"""Compare fixed transposition with offline soft-knee F0 compression."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Unconverted dry audio file")
    parser.add_argument("--model", default="CreatorV2.pth")
    parser.add_argument("--output", help="New output directory; must not exist")
    parser.add_argument("--speaker-id", type=int)
    parser.add_argument("--pitch", type=int, default=11)
    parser.add_argument("--lower-pitch", type=int, default=8)
    parser.add_argument("--threshold-hz", type=float, default=450,
                        help="Experimental output-F0 threshold, NOT a model limit")
    parser.add_argument("--ratio", type=float, default=2)
    parser.add_argument("--knee-semitones", type=float, default=4)
    parser.add_argument("--smoothing-ms", type=float, default=30)
    parser.add_argument("--rms-mix-rate", type=float, default=0.5)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_pipeline_class(base):
    import numpy as np
    from tools.pitch_compression import compress_f0, encode_f0

    class ExperimentPipeline(base):
        def __init__(self, sample_rate, config, settings):
            super().__init__(sample_rate, config)
            self.settings = settings
            self.dynamic = False
            self.cached_input = None
            self.cached_f0 = None
            self.trace = None

        def get_f0(self, x, p_len, f0_up_key, f0_method):
            if f0_method != "rmvpe":
                raise ValueError("This experiment supports RMVPE only")
            if self.cached_input is None or not np.array_equal(x, self.cached_input):
                import torch

                # First-time RMVPE initialization must not change synthesis noise.
                with torch.random.fork_rng():
                    _, extracted = super().get_f0(x, p_len, 0, f0_method)
                self.cached_input = x.copy()
                self.cached_f0 = extracted.copy()
            source = self.cached_f0.copy()
            fixed = source.copy()
            fixed *= 2 ** (f0_up_key / 12)
            output = fixed.copy()
            reduction = np.zeros_like(fixed)
            if self.dynamic:
                output, reduction = compress_f0(
                    fixed, self.settings.threshold_hz, self.settings.ratio,
                    self.settings.knee_semitones, self.settings.smoothing_ms,
                    self.window / self.sr,
                )
                output = output.astype(fixed.dtype)
            if not np.isfinite(output).all():
                raise ValueError("Non-finite F0 produced")
            self.trace = (source, fixed, output, reduction)
            return encode_f0(output), output

    return ExperimentPipeline


def save_trace(path, pipeline, input_samples):
    import numpy as np

    source, fixed, output, reduction = pipeline.trace
    hop = pipeline.window / pipeline.sr
    start = pipeline.t_pad // pipeline.window
    count = min(input_samples // pipeline.window, len(source) - start)
    if count <= 0:
        raise ValueError("No unpadded F0 frames")
    section = slice(start, start + count)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_seconds", "source_f0_interpolated_hz",
                         "fixed_f0_hz", "output_f0_hz", "reduction_semitones"))
        writer.writerows(zip(np.arange(count) * hop, source[section],
                             fixed[section], output[section], reduction[section]))
    positive = output[section][output[section] > 0]
    return {
        "frames": count,
        "modified_frame_fraction": float(np.mean(reduction[section] > 0.01)),
        "max_reduction_semitones": float(np.max(reduction[section])),
        "output_f0_p50_p95_p99_hz": (
            np.percentile(positive, [50, 95, 99]).tolist() if positive.size else []
        ),
        "above_1100hz_frame_fraction": float(np.mean(output[section] > 1100)),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Resolve caller-relative paths before cli changes the working directory.
    input_path = Path(args.input).expanduser().resolve()
    model_value = args.model
    if Path(model_value).expanduser().is_file():
        model_value = str(Path(model_value).expanduser().resolve())
    output_path = (Path(args.output).expanduser().resolve() if args.output else
                   ROOT / "logs" / "f0_experiments" /
                   (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]))
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError("Use a new output directory: %s" % output_path)
    if not 0 <= args.rms_mix_rate <= 1 or not 0 <= args.protect <= 0.5:
        raise ValueError("Require RMS mix in [0,1] and protect in [0,0.5]")
    if not 0 <= args.seed < 2 ** 32:
        raise ValueError("Seed must be in [0, 2**32)")

    os.environ["RVC_CUDA_GRAPH"] = "0"
    from infer.cli import create_config, resolve_model, load_model_metadata, select_speaker
    from tools.pitch_compression import compress_f0
    import numpy as np
    import soundfile as sf
    import torch
    from infer.audio import load_audio
    from infer.vc.modules import VC
    from infer.vc.pipeline import Pipeline
    from infer.vc.utils import load_hubert

    compress_f0(np.zeros(1), args.threshold_hz, args.ratio,
                args.knee_semitones, args.smoothing_ms)
    model_path = resolve_model(model_value)
    os.environ["weight_root"] = str(model_path.parent)
    speaker_id = select_speaker(*load_model_metadata(model_path), args.speaker_id)
    config = create_config()
    vc = VC(config)
    vc.get_vc(model_path.name)
    if vc.if_f0 != 1:
        raise ValueError("Model must support F0 guidance")
    vc.hubert_model = load_hubert(config)
    pipeline = make_pipeline_class(Pipeline)(vc.tgt_sr, config, args)
    vc.pipeline = pipeline
    decoded = load_audio(str(input_path), 16000)
    if decoded.size < 1600 or not np.isfinite(decoded).all():
        raise ValueError("Input must contain at least 0.1 seconds of finite audio")
    if np.max(np.abs(decoded)) < 1e-7:
        raise ValueError("Input is silent; use speech for this experiment")
    output_path.mkdir(parents=True, exist_ok=False)
    report = {
        "settings": vars(args), "input": str(input_path), "model": str(model_path),
        "input_sha256": file_hash(input_path), "model_sha256": file_hash(model_path),
        "speaker_id": speaker_id, "device": str(config.device),
        "precision": str(config.dtype), "index_rate": 0,
        "input_duration_seconds": decoded.size / 16000,
        "decoded_mono_16k_peak": float(np.max(np.abs(decoded))),
        "decoded_mono_16k_near_fullscale_fraction": float(np.mean(np.abs(decoded) >= 0.999)),
        "notes": [
            "F0 is the original pipeline's interpolated conditioning, not raw voiced detection.",
            "Threshold is experimental, not a measured model-safe upper limit.",
            "Compression is offline/noncausal; it is not octave correction or emotion detection.",
            "Identical seeds reduce synthesis variation; bitwise determinism is not guaranteed.",
            "Elapsed time includes warmup and is not a fair realtime performance comparison.",
            "RMS matching is not perceptual loudness matching; F0 metrics are not quality scores.",
        ], "variants": {},
    }
    outputs = {}
    for label, pitch, dynamic in (("fixed", args.pitch, False),
                                  ("lower_fixed", args.lower_pitch, False),
                                  ("dynamic", args.pitch, True)):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        pipeline.dynamic = dynamic
        start = perf_counter()
        status, result = vc.vc_single(speaker_id, str(input_path), pitch, "rmvpe",
                                      "", 0, 0, args.rms_mix_rate, args.protect)
        elapsed = perf_counter() - start
        if not result or result[0] is None or result[1] is None:
            raise RuntimeError("%s inference failed: %s" % (label, status))
        sr, audio = result
        audio = np.asarray(audio)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Invalid synthesized audio: %s" % label)
        sf.write(str(output_path / (label + ".wav")), audio, sr, subtype="PCM_16")
        floating = audio.astype(np.float64) / 32768
        outputs[label] = (sr, floating)
        metrics = save_trace(output_path / (label + "_f0.csv"), pipeline, decoded.size)
        metrics.update(pitch=pitch, elapsed_seconds=elapsed,
                       output_peak=float(np.max(np.abs(floating))),
                       output_rms=float(np.sqrt(np.mean(floating ** 2))))
        report["variants"][label] = metrics
        print("Completed: %s" % label, flush=True)

    # Attenuate to the quietest variant's RMS; apply one shared headroom factor.
    target = min(v["output_rms"] for v in report["variants"].values())
    if target <= 1e-8:
        raise ValueError("A synthesized variant is silent; RMS comparison is invalid")
    matched = {label: audio * target / report["variants"][label]["output_rms"]
               for label, (_, audio) in outputs.items()}
    shared = min(1.0, 0.95 / max(np.max(np.abs(a)) for a in matched.values()))
    for label, audio in matched.items():
        sf.write(str(output_path / (label + "_rms_matched.wav")),
                 audio * shared, outputs[label][0], subtype="PCM_16")
    report["matched_rms"] = target * shared
    (output_path / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("Experiment saved: %s" % output_path, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print("f0-experiment: error: %s" % error, file=sys.stderr)
        raise SystemExit(1)
