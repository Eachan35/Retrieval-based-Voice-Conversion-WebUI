import ast
from pathlib import Path
import unittest

import numpy as np
import torch

from tools.pitch_compression import (
    PitchCompressionSettings, compress_f0_causal,
)


def cache_engine(settings, length=1024):
    path = Path(__file__).resolve().parents[1] / "infer" / "rtrvc.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "RVC")
    methods = [node for node in source.body if isinstance(node, ast.FunctionDef)
               and node.name in ("get_f0_post", "update_pitch_cache", "reset_pitch_cache")]
    cls = ast.ClassDef(name="CacheEngine", bases=[], keywords=[], body=methods,
                       decorator_list=[])
    namespace = {"torch": torch, "PitchCompressionSettings": PitchCompressionSettings,
                 "compress_f0_causal": compress_f0_causal}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(path), "exec"), namespace)
    engine = namespace["CacheEngine"]()
    engine.device = "cpu"
    engine.formant_shift = 0
    engine.f0_mel_min = 1127 * np.log(1 + 50 / 700)
    engine.f0_mel_max = 1127 * np.log(1 + 1100 / 700)
    engine.pitch_compression = settings
    engine.cache_pitch = torch.zeros(length, dtype=torch.long)
    engine.cache_pitchf = torch.zeros(length)
    engine.cache_pitchf_raw = torch.zeros(length)
    return engine


class RealtimePitchTests(unittest.TestCase):
    def test_settings_old_config_and_roundtrip(self):
        settings = PitchCompressionSettings.from_mapping({"pitch": 11})
        self.assertEqual(settings, PitchCompressionSettings())
        self.assertEqual(PitchCompressionSettings.from_mapping(settings.to_mapping()), settings)
        with self.assertRaises(ValueError):
            PitchCompressionSettings.from_mapping({"dynamic_f0_ratio": 0})

    def test_causality(self):
        settings = PitchCompressionSettings()
        prefix = np.linspace(100, 900, 100)
        a, _ = compress_f0_causal(np.r_[prefix, np.full(100, 1200)], settings)
        b, _ = compress_f0_causal(np.r_[prefix, np.full(100, 100)], settings)
        np.testing.assert_array_equal(a[:100], b[:100])

    def test_silence_resets_smoothing(self):
        output, reduction = compress_f0_causal(
            np.r_[np.full(20, 900), np.zeros(5), np.full(20, 100)],
            PitchCompressionSettings(),
        )
        np.testing.assert_array_equal(output[20:25], 0)
        np.testing.assert_array_equal(reduction[25:], 0)

    def test_disabled_matches_original_cache(self):
        engine = cache_engine(PitchCompressionSettings(enabled=False))
        coarse, continuous = engine.get_f0_post(torch.linspace(100, 900, 31))
        expected_coarse = engine.cache_pitch.clone()
        expected_continuous = engine.cache_pitchf.clone()
        for _ in range(5):
            expected_coarse[:-22] = expected_coarse[22:].clone()
            expected_continuous[:-22] = expected_continuous[22:].clone()
            expected_coarse[4 - len(coarse):] = coarse[3:-1]
            expected_continuous[4 - len(coarse):] = continuous[3:-1]
            engine.update_pitch_cache(coarse, continuous, 22)
            self.assertTrue(torch.equal(engine.cache_pitch, expected_coarse))
            self.assertTrue(torch.equal(engine.cache_pitchf, expected_continuous))

    def test_identity_and_reset(self):
        disabled = cache_engine(PitchCompressionSettings(enabled=False))
        enabled = cache_engine(PitchCompressionSettings(ratio=1))
        coarse, continuous = enabled.get_f0_post(torch.linspace(100, 900, 31))
        for engine in (disabled, enabled):
            engine.update_pitch_cache(coarse, continuous, 22)
        self.assertTrue(torch.equal(disabled.cache_pitch, enabled.cache_pitch))
        self.assertTrue(torch.equal(disabled.cache_pitchf, enabled.cache_pitchf))
        enabled.reset_pitch_cache()
        for cache in (enabled.cache_pitch, enabled.cache_pitchf, enabled.cache_pitchf_raw):
            self.assertEqual(torch.count_nonzero(cache).item(), 0)

    def test_overlap_not_recompressed_and_history_continuity(self):
        settings = PitchCompressionSettings()
        engine = cache_engine(settings)
        track = np.linspace(150, 1000, 400)
        history = np.zeros(1024)
        for end in range(40, 400, 22):
            window = torch.tensor(track[end - 31:end], dtype=torch.float32)
            coarse, continuous = engine.get_f0_post(window)
            engine.update_pitch_cache(coarse, continuous, 22)
            history[:-22] = history[22:]
            history[-27:] = window[3:-1].numpy()
            expected, _ = compress_f0_causal(history, settings)
            np.testing.assert_allclose(engine.cache_pitchf.numpy(), expected, atol=0.0001)
            np.testing.assert_array_equal(engine.cache_pitchf_raw.numpy(), history)
        self.assertLess(engine.cache_pitchf[-1], engine.cache_pitchf_raw[-1])

    def test_formant_threshold_compensation(self):
        engine = cache_engine(PitchCompressionSettings(smoothing_ms=0))
        engine.formant_shift = 12
        coarse, continuous = engine.get_f0_post(torch.full((31,), 450.))
        engine.update_pitch_cache(coarse, continuous, 22)
        expected, _ = compress_f0_causal(
            engine.cache_pitchf_raw.numpy(),
            PitchCompressionSettings(threshold_hz=225, smoothing_ms=0),
        )
        np.testing.assert_allclose(engine.cache_pitchf.numpy(), expected, atol=0.0001)


if __name__ == "__main__":
    unittest.main()
