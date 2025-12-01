# tempo.py
# Module for tempo detection from audio files or raw audio data.
# Uses librosa for audio processing.
# Returns estimated tempo in BPM and beat times in seconds.

import librosa
import numpy as np
from typing import Tuple, List, Optional

# detect_tempo returns estimated tempo and beat times from audio.
# It can take either raw audio (y, sr) or a file path.
# It uses librosa for audio processing.
# Returns tempo in BPM and a list of beat times in seconds.

def detect_tempo(
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    path: Optional[str] = None,
    target_sr: Optional[int] = None,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    tightness: float = 100.0,
    aggregate: str = "median",
) -> Tuple[float, List[float]]:

    # Returns (tempo_bpm, beat_times_seconds).
    # - Pass either (y,sr) or a file path.
    # - If target_sr is set, audio is resampled to that rate.
    if y is None:
        if path is None:
            raise ValueError("Provide either (y, sr) or a file path.")
        y, sr = librosa.load(path, sr=target_sr)

    # Onset envelope
    oenv = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, aggregate=aggregate
    )

    # Beat tracking
    tempo_bpm, beat_frames = librosa.beat.beat_track(
        onset_envelope=oenv,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=tightness,
        trim=False,
    )

    # Convert beat frames to time (seconds)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return float(tempo_bpm), beat_times.tolist()

# Convenience wrapper for file paths.
def detect_tempo_from_file(
    path: str,
    target_sr: Optional[int] = None,
    **kwargs) -> Tuple[float, List[float]]:
    #Convenience wrapper for file paths
    return detect_tempo(path=path, target_sr=target_sr, **kwargs)

