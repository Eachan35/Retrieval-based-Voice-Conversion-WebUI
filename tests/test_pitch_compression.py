import unittest
from types import SimpleNamespace

import numpy as np

from tools.pitch_compression import compress_f0, encode_f0
from tools.offline_f0_experiment import make_pipeline_class


class PitchCompressionTests(unittest.TestCase):
    def test_identity_and_silence(self):
        f0 = np.array([0, 100, 450, 900], dtype=np.float32)
        output, reduction = compress_f0(f0, 450, ratio=1)
        np.testing.assert_array_equal(output, f0)
        np.testing.assert_array_equal(reduction, 0)
        output, _ = compress_f0(np.zeros(20), 450)
        np.testing.assert_array_equal(output, 0)

    def test_hard_knee_half_octave(self):
        output, reduction = compress_f0(np.array([225, 450, 900]), 450,
                                       knee_semitones=0, smoothing_ms=0)
        np.testing.assert_allclose(output, [225, 450, 450 * np.sqrt(2)])
        np.testing.assert_allclose(reduction, [0, 0, 6])

    def test_soft_knee_monotonic(self):
        f0 = np.geomspace(50, 2000, 10000)
        output, reduction = compress_f0(f0, 450, smoothing_ms=0)
        self.assertTrue((np.diff(output) > 0).all())
        self.assertTrue((output <= f0).all())
        self.assertTrue((reduction >= 0).all())
        np.testing.assert_array_equal(output[f0 < 350], f0[f0 < 350])

    def test_smoothing_does_not_bridge_unvoiced(self):
        f0 = np.r_[np.full(20, 900), np.zeros(5), np.full(20, 100)]
        output, reduction = compress_f0(f0, 450)
        np.testing.assert_array_equal(output[20:25], 0)
        np.testing.assert_array_equal(reduction[25:], 0)
        self.assertLess(np.diff(compress_f0(np.r_[np.full(20, 100),
                                                 np.full(20, 900)], 450)[1]).max(), 6)

    def test_validation(self):
        for values in ([np.nan], [-1], [[100]]):
            with self.assertRaises(ValueError):
                compress_f0(values, 450)
        for options in ({"threshold_hz": 0}, {"threshold_hz": 450, "ratio": .5},
                        {"threshold_hz": 450, "smoothing_ms": -1},
                        {"threshold_hz": float("inf")}):
            with self.assertRaises(ValueError):
                compress_f0([100], **options)

    def test_coarse_range(self):
        coarse = encode_f0(np.array([0., 50., 1100., 2000.]))
        np.testing.assert_array_equal(coarse, [1, 1, 255, 255])

    def test_pipeline_cache_and_both_pitch_representations(self):
        import torch

        class FakePipeline:
            def __init__(self, sr, config):
                self.sr, self.window, self.calls = 16000, 160, 0

            def get_f0(self, x, p_len, key, method):
                self.calls += 1
                torch.rand(1)
                return None, np.array([100., 450., 900.], dtype=np.float32)

        settings = SimpleNamespace(threshold_hz=450, ratio=2,
                                   knee_semitones=0, smoothing_ms=0)
        pipeline = make_pipeline_class(FakePipeline)(40000, None, settings)
        audio = np.ones(480)
        rng_state = torch.get_rng_state().clone()
        coarse, fixed = pipeline.get_f0(audio, 3, 12, "rmvpe")
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
        np.testing.assert_array_equal(fixed, [200, 900, 1800])
        np.testing.assert_array_equal(coarse, encode_f0(fixed))
        pipeline.dynamic = True
        coarse, dynamic = pipeline.get_f0(audio, 3, 12, "rmvpe")
        self.assertEqual(pipeline.calls, 1)
        self.assertLess(dynamic[-1], fixed[-1])
        np.testing.assert_array_equal(coarse, encode_f0(dynamic))
        pipeline.get_f0(audio * 2, 3, 12, "rmvpe")
        self.assertEqual(pipeline.calls, 2)


if __name__ == "__main__":
    unittest.main()
