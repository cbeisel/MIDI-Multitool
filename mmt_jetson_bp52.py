# Midi Multitool — Jetson Orin Nano (pure ONNX with stitching, no TensorFlow)
#
# Listens for UDP audio packets, saves WAV, runs Basic-Pitch Tensorflow or Onnx models with stitching,
# decodes notes, saves MIDI/JSON, and publishes MQTT feedback.
# Requires onnxruntime with CUDA providers, TensorRT optional, not currently build for the onnxruntime wheel.
# Can also use the TensorFlow Basic-Pitch pipeline if desired.
#
# Tested on Jetson Orin Nano 8GB with JetPack 5.1.1, Python 3.10, ONNX Runtime 1.15.1, Tensorflow 2.16.1+nv24.8
#
# Run from /home/cbeisel/uconniot/
#
# Usage:
#  python3 mmt_jetson_bp5.py [--backend {onnx,tf}] [--test-tone | --test-chord]
#
# Args:
#  --backend {onnx,tf}   Backend to use:
#                        onnx (default): ONNX Runtime + stitched windows
#                        tf            : Basic-Pitch TensorFlow pipeline
#
#  --test-tone           Generate a single 440 Hz tone (exact MODEL_SAMPLES) and run pipeline
#  --test-chord          Generate a chord (A4 + C#5) and run pipeline

from asyncio import events
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # make tf.keras use the legacy (v2) shim
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet logs (optional)
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # allocator grows on demand
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")        # reduce upfront CUDA mem use

import math, sys, json, time, socket, wave, threading, pathlib, argparse
import numpy as np
import librosa, soundfile as sf
import mido
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion # for newer API
from tempo import detect_tempo_from_file
import time
import scipy.signal as sps
from perf_metrics import PerfLogger

# Performance logger
perf = PerfLogger("/home/cbeisel/mmt_logs/pipeline_metrics.csv")

# Argument Parsing
parser = argparse.ArgumentParser(description="MMT Jetson ONNX Audio-to-MIDI Service")
parser.add_argument("--backend", choices=["onnx", "tf"], default="onnx",
                    help="Use ONNX stitching pipeline (default) or TF Basic-Pitch")
parser.add_argument("--test-tone", action="store_true",
                    help="Generate a short test tone and run the pipeline")
parser.add_argument("--test-chord", action="store_true",
                    help="Generate a short test chord and run the pipeline")
args = parser.parse_args()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Configuration
CONFIG_PATH = pathlib.Path("/home/cbeisel/uconniot/mmt_config.json")

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

CFG = load_config()

# Networking / IO
MQTT_BROKER = CFG.get("MQTT_BROKER", "192.168.0.150")
UDP_PORT     = CFG.get("UDP_PORT", 5006)

# Audio / model SR
SAMPLE_RATE = CFG.get("SAMPLE_RATE", 22050)
BYTES_PER_SAMPLE = 2
CHANNELS = 1

# Model & Onnx Runtime paths (with TensorRT support planned)
ONNX_MODEL = "/home/cbeisel/uconniot/lib/python3.10/site-packages/basic_pitch/saved_models/icassp_2022/nmp.onnx"
OPT_ONNX_OUT = "/home/cbeisel/models/nmp_optimized.onnx"  # ORT-optimized graph output
TRT_CACHE = "/home/cbeisel/models/trt_cache"              # TRT engine cache dir

# Basic-Pitch ONNX expects exactly 43844 samples (~1.99 s @ 22.05 kHz)
# (Found this out the hard way)
MODEL_SAMPLES = 43844

# 50% overlap between windows to stitch
HOP_SAMPLES = MODEL_SAMPLES // 2

# Decoding defaults (from mmt_config.json or hardcoded)
ONSET_THRESHOLD         = float(CFG.get("ONSET_THRESHOLD", 0.45))
FRAME_THRESHOLD         = float(CFG.get("FRAME_THRESHOLD", 0.35))
FRAME_HYSTERESIS_RATIO  = float(CFG.get("FRAME_HYSTERESIS_RATIO", 0.8))
MIN_NOTE_SECONDS        = float(CFG.get("MIN_NOTE_SECONDS", 0.05))
MIN_GAP_SECONDS         = float(CFG.get("MIN_GAP_SECONDS", 0.06))
MIN_AMP                 = float(CFG.get("MIN_AMPLITUDE", 0.2))
VEL_MIN                 = int(CFG.get("VELOCITY_MIN", 20))
VEL_MAX                 = int(CFG.get("VELOCITY_MAX", 127))
QUANTIZE_MODE           = CFG.get("QUANTIZE_MODE", "tempo_subdiv")  # off | seconds | tempo_subdiv
QUANTIZE_SECONDS        = float(CFG.get("QUANTIZE_SECONDS", 0.125))
QUANTIZE_SUBDIV         = int(CFG.get("QUANTIZE_SUBDIV", 4))        # 4=16ths, 8=32nds, etc.

# Preprocess options
AUDIO_TRIM_DB           = float(CFG.get("AUDIO_TRIM_DB", 25))
PREEMPHASIS             = float(CFG.get("PREEMPHASIS", 0.97))
NORMALIZE               = bool(CFG.get("NORMALIZE", True))

# MIDI defaults
MIN_MIDI                = 36
MAX_MIDI                = 96
ASSUME_BPM              = float(CFG.get("ASSUME_BPM", 120))
TICKS_PER_BEAT          = int(CFG.get("TICKS_PER_BEAT", 480))

# Capture directory
CAPTURE_DIR = pathlib.Path("/home/cbeisel/mmt_captures")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# Globals
recording = False
frames = []
sock = None
udp_thread = None
wav_filename = None
mqtt_client = None
ort_sess = None

# TensorFlow GPU report
# Prints TensorFlow GPU status at startup
def tf_backend_report_startup():
    try:
        import tensorflow as tf
        from tensorflow import config as tf_config, sysconfig as tf_sysconfig

        # Physical devices
        try:
            phys_gpus = tf_config.list_physical_devices("GPU")
        except Exception as e:
            print(f"[WARN] Could not list TF physical GPUs: {e}")
            phys_gpus = []
        if phys_gpus:
            print(f"[INFO] TensorFlow sees {len(phys_gpus)} physical GPU(s):")
            for d in phys_gpus:
                print(f"       • {d.name}")

            # Configure GPU memory
            logical_configured = False
            try:
                # Only configure if not already configured
                existing_cfg = tf_config.get_logical_device_configuration(phys_gpus[0])
                if not existing_cfg:
                    tf_config.set_logical_device_configuration(
                        phys_gpus[0],
                        [tf_config.LogicalDeviceConfiguration(memory_limit=3072)]  # 3 GB limit
                    )
                logical_configured = True
            except (AttributeError, RuntimeError, ValueError) as e:
                # Older TF, or GPUs already initialized, or other issue
                print(f"[WARN] set_logical_device_configuration skipped: {e}")

            # Set memory growth if logical config not done
            if not logical_configured:
                try:
                    for g in phys_gpus:
                        tf_config.experimental.set_memory_growth(g, True)
                except Exception as e:
                    print(f"[WARN] set_memory_growth failed: {e}")

        else:
            print("[INFO] TensorFlow reports NO GPU devices; running on CPU.")

        # Build info (helps confirm CUDA-enabled wheel installed)
        try:
            bi = tf_sysconfig.get_build_info()
            cuda_ver  = bi.get("cuda_version", "unknown")
            cudnn_ver = bi.get("cudnn_version", "unknown")
            print(f"[INFO] TensorFlow build: CUDA {cuda_ver}, cuDNN {cudnn_ver}")
        except Exception:
            pass

        # Logical devices AFTER config
        try:
            logical_gpu = tf_config.list_logical_devices("GPU")
            if logical_gpu:
                print(f"[INFO] Logical GPU(s): {[d.name for d in logical_gpu]}")
        except Exception as e:
            print(f"[WARN] Could not list logical GPUs: {e}")

    except Exception as e:
        print(f"[WARN] TensorFlow startup attempt failed: {e}")

# OnnxRuntime Session (order of priority: TensorRT->CUDA->CPU)
# TensorRT requires onnxruntime built with TensorRT support, this is still on the to-do list, installed OnnxRuntime wheel does not have it.
# Need to still refine the Onnx model for acceptable results, using the TF pipeline is preferred at this point.
def init_ort_session():
    import onnxruntime as ort
    print("[INFO] Initializing ONNX Runtime with providers: ['TensorrtExecutionProvider','CUDAExecutionProvider','CPUExecutionProvider']")
    
    # Ensure cache directories exist
    pathlib.Path(TRT_CACHE).mkdir(parents=True, exist_ok=True)
    pathlib.Path(OPT_ONNX_OUT).parent.mkdir(parents=True, exist_ok=True)

    # env vars (optional) — convenient if you later run under systemd
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1")
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_PATH", TRT_CACHE)
    os.environ.setdefault("ORT_TENSORRT_FP16_ENABLE", "1")

    # Session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Save an ORT-optimized model (hardware-specific optimizations are fine on the same Jetson)
    sess_options.optimized_model_filepath = OPT_ONNX_OUT

    # Provider priority: TensorRT > CUDA > CPU
    # TensorRT needs onnxruntime built with TensorRT support (not yet implemented in the wheel)
    providers = [
        ("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": TRT_CACHE,
            "trt_fp16_enable": True
        }),
        ("CUDAExecutionProvider", {}),
        ("CPUExecutionProvider", {})
    ]

    # Create session
    sess = ort.InferenceSession(ONNX_MODEL, sess_options=sess_options, providers=providers)

    # Check providers
    avail = ort.get_available_providers()
    active = sess.get_providers()
    if "TensorrtExecutionProvider" not in avail:
        print("[WARN] TensorRT EP not available; falling back to CUDA/CPU.")
    print(f"[INFO] ORT providers available: {avail}")
    print(f"[INFO] ORT providers active:    {active}")

    # Print input/output info
    for i, o in enumerate(sess.get_outputs()):
        print(f"[INFO] ONNX output[{i}]: name={o.name}, shape={o.shape}")
    return sess

# UDP Audio Listener
# Future improvements, consider transitioning to TCP
# (so far no significant packet loss observed, perfectly acceptable for streaming audio).
def udp_listener():
    global recording, frames, sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)
    print(f"[UDP] Listening on port {UDP_PORT}...")
 
    # Listen loop
    while recording:
        try:
            # Receive UDP packet
            data, _ = sock.recvfrom(4096)
            # Append to frames
            frames.append(data)
        except socket.timeout:
            # Timeout, just continue
            pass
        except Exception as e:
            # Other socket error
            print(f"[UDP] Error: {e}")
            break
    if sock:
        sock.close()


# MQTT Command Handlers
def on_message(client, userdata, msg):
    global recording, frames, udp_thread, wav_filename
    topic, payload = msg.topic, msg.payload.decode()
    print(f"[MQTT] {topic} -> {payload}")

    # Parse JSON command
    try:
        message = json.loads(payload)
    except json.JSONDecodeError:
        print("[WARN] Non-JSON message, ignoring")
        return
    if message.get("type") != "control":
        return

    # Handle commands
    cmd = message.get("command")
    if cmd == "start_recording" and not recording:  # start UDP capture
        recording = True
        frames = []
        ts = int(time.time())
        wav_filename = CAPTURE_DIR / f"recording_{ts}.wav"
        udp_thread = threading.Thread(target=udp_listener, daemon=True)
        udp_thread.start()
        print("[INFO] Recording started")

    elif cmd == "stop_recording" and recording: # stop UDP capture and save WAV
        recording = False
        udp_thread.join()
        print("[INFO] Recording stopped")
        try:
            with wave.open(str(wav_filename), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(BYTES_PER_SAMPLE)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))
            print(f"[INFO] Saved {wav_filename}")
        except Exception as e:
            print(f"[ERROR] Writing WAV failed: {e}")
            return
        run_amt_pipeline(str(wav_filename))
    
    elif cmd == "set_tempo":  # set tempo (need to implement)
        tempo = message.get("tempo")
        print(f"[INFO] Tempo set to {tempo} BPM")

    elif cmd == "set_loop_length":  # set loop length (need to implement)
        loop_length = message.get("loop_length")
        print(f"[INFO] Loop length set to {loop_length} bars")

    elif cmd == "heartbeat":  # heartbeat message
        print("[INFO] Heartbeat received")

    else:
        print(f"[WARN] Unknown or invalid command: {cmd}") # ignore commands not recognized

# MQTT Feedback
def publish_feedback(client, wav_file, cleaned_notes, tempo=None, loop_length=None):
    feedback = {
        "type": "feedback",
        "status": "processed",
        "file": pathlib.Path(wav_file).name,
        "notes_count": len(cleaned_notes),
    }

    # Optional fields, tap tempo or detect tempo/loop length?
    if tempo is not None:
        feedback["tempo"] = tempo
    if loop_length is not None:
        feedback["loop_length"] = loop_length
    if client is not None:
        client.publish("mmt/feedback", json.dumps(feedback))
    print(f"[INFO] Published feedback: {feedback}")

# Highpass filter
def highpass_filter(y, sr, cutoff=70.0, order=4):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = sps.butter(order, normal_cutoff, btype='high', analog=False)
    y_filtered = sps.lfilter(b, a, y)
    return y_filtered

# Lowpass filter
def lowpass_filter(y, sr, cutoff=9000.0, order=4):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = sps.butter(order, normal_cutoff, btype='low', analog=False)
    y_filtered = sps.lfilter(b, a, y)
    return y_filtered

# Audio Preprocessing
# uses librosa to load, preprocess, and save audio
# Some filtering and processing steps are commented out for future tuning
def preprocess_audio(input_wav, output_wav, target_sr=SAMPLE_RATE):
    y, sr = librosa.load(input_wav, sr=None)

    # Pre-emphasis
    if PREEMPHASIS:
        y = librosa.effects.preemphasis(
            y,
            coef=float(PREEMPHASIS),
            zi = 0.0)

    # Highpass filter
    # y = highpass_filter(y, sr, cutoff=70.0)

    # Lowpass filter
    # y = lowpass_filter(y, sr, cutoff=9000.0)

    # Remove DC offset
    y = y - np.mean(y)

    # Normalize (need to consider a softer normalization for very quiet audio)
    # (Also need to build a -12dB attenuator for the input stage on before the ADC on the ESP32-S3)
    if NORMALIZE:
        y = librosa.util.normalize(y)

    # Trim silence
    # y, _ = librosa.effects.trim(y, top_db=float(AUDIO_TRIM_DB))

    # Noise gate
    # gate_threshold = 0.01 * np.max(np.abs(y))
    # y[np.abs(y) < gate_threshold] = 0.0

    # Resample if not target sample rate
    # if sr != target_sr:
    #     y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    #     sr = target_sr

    # Save preprocessed audio
    sf.write(output_wav, y, sr)
    return output_wav

# Head Classifier (frames/onsets/contour)
def classify_heads(raw_outputs):
    # Extract outputs and identify heads by shape
    outs = [np.squeeze(o, axis=0) for o in raw_outputs]   # each (T, P)
    P = [a.shape[1] for a in outs]

    try:
        # Find indices by shape
        contour_idx = P.index(264)               # 264 = contour
        eighty8 = [i for i, p in enumerate(P) if p == 88]  # two heads at 88: frames & onsets
        # pick frames vs onsets by density (frames denser)
        dens = [(outs[i] > 0.5).mean() for i in eighty8] # density of activations
        frames_idx = eighty8[int(np.argmax(dens))]
        onsets_idx = eighty8[int(np.argmin(dens))]
    except ValueError as e:
        raise RuntimeError(f"Unexpected ONNX head widths {P}; expected two 88s and one 264") from e

    # Assign outputs
    frames  = outs[frames_idx]   # (T, 88)
    onsets  = outs[onsets_idx]   # (T, 88)
    contour = outs[contour_idx]  # (T, 264)
    print(f"[INFO] Head mapping (by shape): frames={frames_idx}, onsets={onsets_idx}, contour={contour_idx}")
    return frames, onsets, contour


# Decode Notes (stitched timeline)
# Converts model outputs into note events with start/end times, pitch, and velocity
def decode_notes(frames, onsets, audio_seconds):
    T, P = frames.shape # time frames, pitch bins
    sec_per_frame = audio_seconds / max(T, 1)

    # Map bins to MIDI
    if P == 128:
        midi_for_bin = np.arange(128)
    elif P == (MAX_MIDI - MIN_MIDI + 1):
        midi_for_bin = np.arange(MIN_MIDI, MAX_MIDI + 1)
    else:
        midi_for_bin = np.linspace(MIN_MIDI, MAX_MIDI, num=P).round().astype(int)

    # Start when frames > hi
    frame_hi = FRAME_THRESHOLD          

    # End when frames < lo
    frame_lo = FRAME_THRESHOLD * FRAME_HYSTERESIS_RATIO
    
    # Min silence between two notes on same pitch
    min_gap_s = MIN_GAP_SECONDS

    active      = np.zeros(P, dtype=bool) # currently active notes
    start_idx   = np.zeros(P, dtype=int) # start frame index per pitch
    max_amp     = np.zeros(P, dtype=float) # peak amplitude per pitch
    last_off_s  = np.full(P, -1e9, dtype=float)  # last note-off time per pitch

    notes = [] # decoded note events

    # Main decoding loop
    for t in range(T):
        f_t = frames[t] # frame activations at time t
        o_t = onsets[t] if onsets is not None else f_t # onset activations at time t

        above_hi = f_t > frame_hi # frames above high threshold
        above_lo = f_t > frame_lo # frames above low threshold
        onset_now = o_t > ONSET_THRESHOLD # onsets above threshold

        # End notes (hysteresis: drop below low threshold)
        ended = active & (~above_lo)
        if ended.any():
            for p_idx in np.where(ended)[0]:
                s = start_idx[p_idx]; e = t
                dur_s = (e - s) * sec_per_frame
                if dur_s >= MIN_NOTE_SECONDS:
                    t0 = s * sec_per_frame
                    t1 = e * sec_per_frame
                    notes.append((t0, t1, int(midi_for_bin[p_idx]), float(max_amp[p_idx])))
                    last_off_s[p_idx] = t1
                active[p_idx] = False
                max_amp[p_idx] = 0.0

        # Start notes: need frames above_hi AND (onset or rising edge), plus debounce
        starters = (~active) & above_hi & (onset_now)
        if starters.any():
            for p_idx in np.where(starters)[0]:
                # Debounce: require some silence since last-off
                tsec = t * sec_per_frame
                if (tsec - last_off_s[p_idx]) < min_gap_s:
                    continue
                active[p_idx] = True
                start_idx[p_idx] = t
                max_amp[p_idx] = f_t[p_idx]

        # Update peak for sustained notes
        holding = active & above_lo
        if holding.any():
            idx = np.where(holding)[0] 
            max_amp[idx] = np.maximum(max_amp[idx], f_t[idx])

    # Flush any still-active notes
    if active.any():
        for p_idx in np.where(active)[0]:
            s = start_idx[p_idx]; e = T
            dur_s = (e - s) * sec_per_frame
            if dur_s >= MIN_NOTE_SECONDS:
                notes.append((s*sec_per_frame, e*sec_per_frame,
                              int(midi_for_bin[p_idx]), float(max_amp[p_idx])))

    return notes

# Note Cleaning + Quantization
def clean_notes(
    note_events,
    min_duration=MIN_NOTE_SECONDS,
    min_amp=MIN_AMP,
    velocity_range=(VEL_MIN, VEL_MAX),
    quantize_grid=None
):
    cleaned, last_note_by_pitch = [], {}
    for n in note_events: # iterate through note events
        try:
            start_time, end_time, pitch, amplitude = n # unpack note event
        except Exception:
            continue
        dur = end_time - start_time # duration in seconds
 
        # Filter by duration and amplitude
        if dur < min_duration or amplitude < min_amp:
            continue

        # Quantize start/end times
        if quantize_grid and quantize_grid > 0:
            start_time = round(start_time / quantize_grid) * quantize_grid
            end_time = round(end_time / quantize_grid) * quantize_grid
            if end_time <= start_time:
                end_time = start_time + quantize_grid
        
        # Map amplitude to velocity
        velocity = int(np.interp(np.sqrt(max(0.0, float(amplitude))), [0, 1], velocity_range))

        # Merge only if quantization is enabled
        if quantize_grid and pitch in last_note_by_pitch:
            last_note = last_note_by_pitch[pitch]
            if abs(start_time - last_note['end']) <= quantize_grid / 2.0:
                last_note['end'] = end_time
                last_note['velocity'] = max(last_note['velocity'], velocity)
                continue

        # Append cleaned note
        note = {"start": float(start_time), "end": float(end_time),
                "pitch": int(pitch), "velocity": velocity}
        cleaned.append(note)
        last_note_by_pitch[pitch] = note
    return cleaned

# MIDI writer, converts note events into a MIDI file
def notes_to_midi(notes, out_path, bpm=ASSUME_BPM, ticks_per_beat=TICKS_PER_BEAT):
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack(); mid.tracks.append(track)
    tempo = mido.bpm2tempo(bpm)
    events = []

    # Build note on/off events
    for n in notes:
        start, end, pitch, vel = n["start"], n["end"], n["pitch"], n["velocity"]
        events.append(("on",  start, pitch, vel))
        events.append(("off", end,   pitch, 0))
    events.sort(key=lambda x: (x[1], 0 if x[0]=="on" else 1))
    cur = 0.0
    for kind, tsec, pitch, vel in events:
        delta = int(mido.second2tick(max(0.0, tsec - cur), ticks_per_beat, tempo))
        track.append(mido.Message("note_on" if kind=="on" else "note_off",
                                  note=int(pitch), velocity=int(vel), time=delta))
        cur = tsec
    mid.save(str(out_path))

# Convert cleaned notes to event list
def cleaned_to_events(cleaned):
    events = []
    for n in cleaned:
        s = n["start"]
        e = n["end"]
        p = n["pitch"]
        v = n["velocity"]
        events.append((s, 0x90, p, v))
        events.append((e, 0x80, p, 0))
    events.sort(key=lambda x: (x[0], 0 if x[1]==0x90 else 1))
    return events

# Publish cleaned note events over MQTT
def publish_note_events_mqtt(client, cleaned_notes, tempo=None,
                              topic="mmt/note_events"):
    # Check client
    if client is None:
        print("[WARN] MQTT client not available; cannot publish note events.")
        return

    # Convert cleaned notes to event list
    ev_list = cleaned_to_events(cleaned_notes)

    # Check for events
    if not ev_list:
        print("[INFO] No note events to publish.")
        return

    # Determine pattern length from last event time
    pattern_length = float(ev_list[-1][0])

    # Build payload
    payload = {
        "type": "note_events",
        "pattern_length": pattern_length,
        "events": [
            {
                "t":float(round(t, 4)),
                "status": int(status),
                "d1": int(d1),
                "d2": int(d2)
            }
            for t, status, d1, d2 in ev_list
        ],
    }

    # Optional tempo
    if tempo is not None:
        try:
            payload["tempo"] = tempo
        except Exception:
            pass

    # Publish to MQTT
    client.publish(topic, json.dumps(payload))
    print(
        f"[INFO] Published {len(ev_list)} note events to MQTT topic '{topic}, "
        f"length={pattern_length:.3f}s"
    )

#TensorFlow Basic-Pitch Inference backend (not used in ONNX mode, called from run_amt_pipeline)
def run_basic_pitch_tf(preprocessed_wav, out_stem):
    import tensorflow as tf
    from basic_pitch.inference import predict

    # TensorFlow GPU info
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[INFO] TensorFlow backend initialized - using {len(gpus)} GPU(s).")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("[INFO] TensorFlow backend initialized - no GPU found, using CPU.")
    except Exception as e:
        print(f"[WARN] TensorFlow GPU initialization error: {e}")

    # Run Basic Pitch TF inference
    model_output, midi_data, note_events = predict(preprocessed_wav)
 
    # Clean and convert note events


    cleaned = []
    try:
        instruments = midi_data.instruments
    except AttributeError:
        # Fallback if midi_data behaves like a mido.MidiFile instead of pretty_midi
        instruments = []

    if instruments:
        # pretty_midi.PrettyMIDI style
        for inst in instruments:
            for n in inst.notes:
                start_time = float(n.start)
                end_time   = float(n.end)
                pitch      = int(n.pitch)
                vel_raw    = int(n.velocity)

                dur = end_time - start_time
                if dur < MIN_NOTE_SECONDS:
                    continue

                # Clamp velocity into your configured range
                velocity = int(np.clip(vel_raw, VEL_MIN, VEL_MAX))

                cleaned.append({
                    "start": start_time,
                    "end":   end_time,
                    "pitch": pitch,
                    "velocity": velocity,
                })
    else:
        # If midi_data isn't pretty_midi, fall back to your old note_events logic
        for ev in note_events:

            try:
                start_time, end_time, pitch, amplitude = ev
            except Exception:
                continue

            amp = float(amplitude)
            amp = max(0.0, min(1.0, amp))
            velocity = int(np.interp(np.sqrt(amp), [0.0, 1.0], [VEL_MIN, VEL_MAX]))
            
            cleaned.append({
                "start": float(start_time),
                "end": float(end_time),
                "pitch": int(pitch),
                "velocity": int(velocity)
            })

    # Write MIDI and JSON outputs
    midi_path = f"{out_stem}_bp.mid"
    json_path = f"{out_stem}_bp.json"
    try:
        midi_data.write(midi_path)
    except AttributeError:
        midi_data.save(midi_path)
    try:
        with open(json_path, "w") as f:
            json.dump(cleaned, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write JSON file: {e}")
    print(f"[INFO] TF Backend MIDI: {midi_path}, JSON: {json_path}")
    print(f"[INFO] TF backend cleaned notes: {len(cleaned)}")
    return midi_path, json_path, cleaned

# Main AMT pipeline to run inference, decoding, cleaning, and saving
def run_amt_pipeline(wav_file):

    print(f"[INFO] Running AMT pipeline on {wav_file}...")
    wav_path = pathlib.Path(wav_file)
    pre_path = wav_path.with_name(f"{wav_path.stem}_pre.wav")

    # Send MQTT processing started message
    if mqtt_client is not None:
        mqtt_client.publish("mmt/feedback", json.dumps({
            "type": "feedback",
            "status": "processing_started",
            "file": wav_path.name
        }))

    # Start timer
    t0 = time.perf_counter()

    # Preprocess audio
    preprocessed = preprocess_audio(str(wav_path), str(pre_path), target_sr=SAMPLE_RATE)
    out_stem = str(wav_path.with_suffix(""))
    print(f"[INFO] Preprocessed audio saved to {preprocessed}")
    
    # Star time for inference
    t1 = time.perf_counter()

    # Run TensorFlow Basic-Pitch backend
    if args.backend == "tf":
            midi_path, json_path, cleaned = run_basic_pitch_tf(preprocessed, out_stem)
            bpm, _ = detect_tempo_from_file(
                str(preprocessed),
                target_sr=SAMPLE_RATE,
                hop_length=512,
                start_bpm=ASSUME_BPM,
                tightness=100.0
            )

            # Post-inference time
            t2 = time.perf_counter()

            # Validate detected BPM
            try:
                detected_bpm = float(bpm)
                if not np.isfinite(detected_bpm):
                    detected_bpm = ASSUME_BPM
            except Exception:
                detected_bpm = ASSUME_BPM
            detected_bpm = max(30.0, min(240.0, detected_bpm))

            # Compute the grid from config
            if QUANTIZE_MODE == "off":
                q_grid = None
            elif QUANTIZE_MODE == "seconds":
                q_grid = QUANTIZE_SECONDS
            else:
                q_grid = 60.0 / (detected_bpm * max(1, QUANTIZE_SUBDIV))

            # Re-clean notes with quantization
            def re_quantize_cleaned( notes,
                                     q):
                if not q:
                    return notes
                out, last_by_pitch = [], {}
                for n in notes:
                    s, e, p, a = n["start"], n["end"], n["pitch"], n["velocity"]
                    s = round(s / q) * q
                    e = round(e / q) * q
                    if e <= s:
                        e = s + q
                    if p in last_by_pitch:
                        last_n = last_by_pitch[p]
                        if abs(s - last_n['end']) <= q / 2.0:
                            last_n['end'] = e
                            last_n['velocity'] = max(last_n['velocity'], a)
                            continue
                    note = {"start": float(s), "end": float(e),
                            "pitch": int(p), "velocity": int(a)}
                    out.append(note)
                    last_by_pitch[p] = note
                return out
            
            # Re-quantize cleaned notes
            cleaned = re_quantize_cleaned(cleaned, q_grid)
            print(f"[INFO] TF Backend MIDI: {midi_path}, JSON: {json_path}")
    else:

        # ONNX Inference backend

        # post-inference time 
        t2 = time.perf_counter()
        
        # Load audio
        y, sr = librosa.load(preprocessed, sr=SAMPLE_RATE)
        total_samples = len(y)
        if total_samples == 0:
            print("[WARN] Empty audio after preprocess.")
            return

        # Number of hops with 50% overlap
        num_hops = max(0, math.floor(max(0, total_samples - MODEL_SAMPLES) / HOP_SAMPLES)) + 1
        in_name = ort_sess.get_inputs()[0].name

        # ---- Probe shapes from first window ----
        start0, end0 = 0, MODEL_SAMPLES
        chunk0 = y[start0:end0]
        if len(chunk0) < MODEL_SAMPLES:
            pad = np.zeros(MODEL_SAMPLES, dtype=np.float32)
            pad[:len(chunk0)] = chunk0.astype(np.float32)
            chunk0 = pad
        else:
            chunk0 = chunk0.astype(np.float32)
        model_input0 = np.expand_dims(chunk0, axis=(0, 2))  # (1, 43844, 1)
        raw_outputs0 = ort_sess.run(None, {in_name: model_input0})
        frames_win0, onsets_win0, _ = classify_heads(raw_outputs0)
        frames_per_window = frames_win0.shape[0]
        P_guess = frames_win0.shape[1]

        # Convert sample overlap to frame overlap
        frames_hop = max(1, round(frames_per_window * (HOP_SAMPLES / MODEL_SAMPLES)))
        total_frames = (num_hops - 1) * frames_hop + frames_per_window

        # Allocate stitchers (floats for averaging overlaps)
        stitched_frames = np.zeros((total_frames, P_guess), dtype=np.float32)
        stitched_onsets = np.zeros((total_frames, P_guess), dtype=np.float32)
        counts = np.zeros((total_frames, P_guess), dtype=np.float32)

        # Sliding inference with overlap-add
        for hop in range(num_hops):
            start = hop * HOP_SAMPLES
            end   = start + MODEL_SAMPLES
            chunk = y[start:end]
            if len(chunk) < MODEL_SAMPLES:
                pad = np.zeros(MODEL_SAMPLES, dtype=np.float32)
                pad[:len(chunk)] = chunk.astype(np.float32)
                chunk = pad
            else:
                chunk = chunk.astype(np.float32)
            
            # Run ONNX model
            model_input = np.expand_dims(chunk, axis=(0, 2))
            raw_outputs = ort_sess.run(None, {in_name: model_input})
            frames_win, onsets_win, _ = classify_heads(raw_outputs)

            # Guard against pitch dimension changes
            if frames_win.shape[1] != P_guess:
                print(f"[WARN] Pitch dim changed {frames_win.shape[1]}→{P_guess}; clipping to min.")
            P_use = min(P_guess, frames_win.shape[1], onsets_win.shape[1])

            # Compute frame range in full timeline
            g0 = hop * frames_hop
            g1 = g0 + frames_per_window
            stitched_frames[g0:g1, :P_use] += frames_win[:, :P_use]
            stitched_onsets[g0:g1, :P_use] += onsets_win[:, :P_use]
            counts[g0:g1, :P_use] += 1.0

        # Average overlaps
        counts[counts == 0] = 1.0
        frames_full = stitched_frames / counts
        onsets_full = stitched_onsets / counts

        # Decode notes from stitched outputs
        audio_seconds = total_samples / float(sr)
        note_events = decode_notes(frames_full, onsets_full, audio_seconds)

        # Detect tempo on the SAME preprocessed audio
        bpm, beat_times = detect_tempo_from_file(
            str(preprocessed),
            target_sr=SAMPLE_RATE,
            hop_length=512,
            start_bpm=ASSUME_BPM,
            tightness=100.0
        )
        try:
            detected_bpm = float(bpm)
            if not np.isfinite(detected_bpm):
                detected_bpm = ASSUME_BPM
        except Exception:
            detected_bpm = ASSUME_BPM
        detected_bpm = max(30.0, min(240.0, detected_bpm))
        print(f"[INFO] Tempo detected: {detected_bpm:.2f} BPM")

        # Compute the grid from config
        if QUANTIZE_MODE == "off":
            q_grid = None
        elif QUANTIZE_MODE == "seconds":
            q_grid = QUANTIZE_SECONDS
        else:
            q_grid = 60.0 / (detected_bpm * max(1, QUANTIZE_SUBDIV))

        # Clean note events
        cleaned = clean_notes(
            note_events,
            min_duration=MIN_NOTE_SECONDS,
            min_amp=MIN_AMP,
            velocity_range=(VEL_MIN, VEL_MAX),
            quantize_grid=q_grid
        )

        print(f"[INFO] Tempo detected: {detected_bpm:.2f} BPM")

    # End timer
    t3 = time.perf_counter()    

    # Report timings
    preprocess_ms = (t1 - t0) * 1000.0
    inference_ms = (t2 - t1) * 1000.0
    postprocess_ms = (t3 - t2) * 1000.0
    total_pipeline_ms = (t3 - t0) * 1000.0

    perf.log_pipeline_metrics(
        wav_file,
        preprocess_ms,
        inference_ms,
        postprocess_ms,
        total_pipeline_ms,
    )

    print(
        f"[PERF] preprocess={preprocess_ms:.1f} ms, "
        f"inference={inference_ms:.1f} ms, "
        f"post={postprocess_ms:.1f} ms, "
        f"total={total_pipeline_ms:.1f} ms"
    )


    midi_path = wav_path.with_suffix(".mid")
    json_path = wav_path.with_suffix(".json")

    # Write MIDI
    try:
          midi_events = cleaned
          notes_to_midi(midi_events, midi_path, bpm=detected_bpm)
    except Exception as e:
        print(f"[ERROR] Writing MIDI failed: {e}")

    # Write JSON
    try:
        with open(json_path, "w") as f:
            json.dump(cleaned, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Writing JSON failed: {e}")

    print(f"[INFO] MIDI: {midi_path}, JSON: {json_path}")

    print(f"[INFO] MIDI: {midi_path}, JSON: {json_path}")
    
    # MQTT publish
    publish_feedback(mqtt_client, wav_file, cleaned, 
                     tempo=detected_bpm, loop_length=4)
    publish_note_events_mqtt(mqtt_client, cleaned,
                             tempo=detected_bpm,
    )

# Test Tone Generator
def make_test_tone(path=str(CAPTURE_DIR / "test_tone.wav"),
                   freq=440.0,
                   sr=SAMPLE_RATE,
                   samples=MODEL_SAMPLES,
                   amp=0.2):
                   
    # Generate exactly MODEL_SAMPLES samples so we exercise one model window.
    t = np.arange(samples) / sr

    # Modest amplitude to avoid normalization edge cases
    y = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, y, sr)
    print(f"[TEST] Wrote {path} ({samples} samples @ {sr} Hz ≈ {samples/sr:.3f}s)")
    return str(path)

# Test chord (same as make_test_tone but multiple freqs)
def make_test_chord(path=str(CAPTURE_DIR / "test_chord.wav"),
                    freqs=(440.0, 554.37),  # A4 + C#5
                    sr=SAMPLE_RATE,
                    samples=MODEL_SAMPLES,
                    amp=0.2):
    t = np.arange(samples)/sr
    y = sum(amp*np.sin(2*np.pi*f*t) for f in freqs).astype(np.float32)
    sf.write(path, y, sr)
    print(f"[TEST] Wrote {path}")
    return str(path)

# Main Loop
def main():

    # Initialize MQTT
    global mqtt_client

    # Initialize ONNX session if needed
    if args.backend == "onnx":
        global ort_sess
        ort_sess = init_ort_session()
    else:
        ort_sess = None  # Not used in TF mode
        print("[INFO] TensorFlow backend selected; skipping ONNX initialization.")

    mqtt_client = mqtt.Client(CallbackAPIVersion.VERSION2)
    mqtt_client.on_message = on_message

    # Connect to MQTT broker
    try:
        mqtt_client.connect(MQTT_BROKER, 1883, 60)
    except Exception as e:
        print(f"[ERROR] MQTT connect failed: {e}")
        sys.exit(1)

    # Subscribe to MQTT topics
    mqtt_client.subscribe([("mmt/audio_control", 2), ("mmt/heartbeat", 2)])
    print("[INFO] Ready. Waiting for MQTT commands...")

    # Start MQTT loopback
    mqtt_client.loop_start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Exiting cleanly...")
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

# Entry Point
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-tone":
        ort_sess = init_ort_session()
        wav = make_test_tone()
        run_amt_pipeline(wav)
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-chord":
        ort_sess = init_ort_session()
        wav = make_test_chord()
        run_amt_pipeline(wav)
        sys.exit(0)
    else:
        if args.backend == "tf":
            os.environ["ORT_DISABLE_MEMORY_ARENAS"] = "1"  # avoid conflicts
            print("[INFO] Running in TensorFlow backend for Basic-Pitch.")
            tf_backend_report_startup()
        else:
            print("[INFO] Running with ONNX stitching pipeline.")
        main()