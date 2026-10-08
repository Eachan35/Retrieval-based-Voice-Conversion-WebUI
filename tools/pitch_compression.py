"""Offline soft-knee compression in log-frequency space."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter
from dataclasses import dataclass


@dataclass(frozen=True)
class PitchCompressionSettings:
    enabled: bool = True
    threshold_hz: float = 450.0
    ratio: float = 2.0
    knee_semitones: float = 4.0
    smoothing_ms: float = 30.0

    def __post_init__(self):
        compress_f0([], self.threshold_hz, self.ratio, self.knee_semitones,
                    self.smoothing_ms)

    @classmethod
    def from_mapping(cls, values):
        defaults = cls()
        return cls(**{
            name: (bool(values.get("dynamic_f0_" + name, getattr(defaults, name)))
                   if name == "enabled" else
                   float(values.get("dynamic_f0_" + name, getattr(defaults, name))))
            for name in cls.__dataclass_fields__
        })

    def to_mapping(self):
        return {"dynamic_f0_" + name: getattr(self, name)
                for name in self.__dataclass_fields__}


def compress_f0_causal(f0, settings, hop_seconds=0.01):
    """Process a RAW history window once, never an already compressed cache."""
    output, reduction = compress_f0(
        f0, settings.threshold_hz, settings.ratio, settings.knee_semitones,
        smoothing_ms=0, hop_seconds=hop_seconds,
    )
    if settings.smoothing_ms == 0 or len(reduction) == 0:
        return output, reduction
    alpha = np.exp(-hop_seconds / (settings.smoothing_ms / 1000))
    voiced = np.asarray(f0) > 0
    boundaries = np.diff(np.r_[False, voiced, False].astype(np.int8))
    for start, end in zip(np.flatnonzero(boundaries == 1),
                          np.flatnonzero(boundaries == -1)):
        reduction[start:end] = lfilter(
            [1 - alpha], [1, -alpha], reduction[start:end]
        )
    return np.asarray(f0) * np.exp2(-reduction / 12), reduction


def compress_f0(f0, threshold_hz, ratio=2.0, knee_semitones=4.0,
                smoothing_ms=30.0, hop_seconds=0.01):
    values = np.asarray(f0, dtype=np.float64)
    parameters = (threshold_hz, ratio, knee_semitones, smoothing_ms, hop_seconds)
    if not all(np.isfinite(value) for value in parameters):
        raise ValueError("Compression parameters must be finite")
    if threshold_hz <= 0 or ratio < 1 or knee_semitones < 0:
        raise ValueError("Require threshold > 0, ratio >= 1 and knee >= 0")
    if smoothing_ms < 0 or hop_seconds <= 0:
        raise ValueError("Require smoothing >= 0 and hop > 0")
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("F0 must be a finite, nonnegative 1-D array")
    reduction = np.zeros_like(values)
    voiced = values > 0
    offset = 12 * np.log2(values[voiced] / threshold_hz)
    if knee_semitones == 0:
        amount = np.maximum(offset, 0)
    else:
        half = knee_semitones / 2
        amount = np.where(
            offset <= -half, 0,
            np.where(offset >= half, offset,
                     (offset + half) ** 2 / (2 * knee_semitones)),
        )
    reduction[voiced] = amount * (1 - 1 / ratio)
    if smoothing_ms > 0:
        # Smooth only the reduction, without connecting separate voiced runs.
        boundaries = np.diff(np.r_[False, voiced, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(boundaries == 1),
                              np.flatnonzero(boundaries == -1)):
            reduction[start:end] = gaussian_filter1d(
                reduction[start:end], smoothing_ms / 1000 / hop_seconds,
                mode="nearest",
            )
    result = values * np.exp2(-reduction / 12)
    return result, reduction


def encode_f0(f0):
    """Match the offline RVC pipeline's 50--1100 Hz coarse encoding."""
    values = np.asarray(f0)
    mel = 1127 * np.log(1 + values / 700)
    lower = 1127 * np.log(1 + 50 / 700)
    upper = 1127 * np.log(1 + 1100 / 700)
    positive = mel > 0
    mel[positive] = (mel[positive] - lower) * 254 / (upper - lower) + 1
    return np.rint(np.clip(mel, 1, 255)).astype(np.int32)
