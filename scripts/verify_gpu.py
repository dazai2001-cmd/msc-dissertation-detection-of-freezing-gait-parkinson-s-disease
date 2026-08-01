"""Verify that the fair benchmark models can execute on a TensorFlow GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fair_benchmark import ARCHITECTURES, BenchmarkSettings, build_benchmark_model
from fog_pipeline import TARGET_COLUMNS


def main() -> None:
    gpu_devices = tf.config.list_physical_devices("GPU")
    if not gpu_devices:
        raise RuntimeError("TensorFlow cannot see a GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(42)

    settings = BenchmarkSettings(native_sampling_rate_hz=100.0, folds_to_run=(0,))
    generator = np.random.default_rng(42)
    x = generator.normal(
        size=(64, settings.window_samples, 4)
    ).astype(np.float32)
    fog = generator.integers(0, 2, size=(64, 1)).astype(np.float32)
    event_type = generator.integers(
        0, 2, size=(64, len(TARGET_COLUMNS))
    ).astype(np.float32)
    weights = np.ones(64, dtype=np.float32)

    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU devices: {gpu_devices}")
    for architecture in ARCHITECTURES:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model = build_benchmark_model(
            architecture,
            fog_positive_weight=1.0,
            type_positive_weights=np.ones(
                len(TARGET_COLUMNS), dtype=np.float32
            ),
            settings=settings,
        )
        metrics = model.train_on_batch(
            x,
            {"fog": fog, "event_type": event_type},
            sample_weight={"fog": weights, "event_type": weights},
            return_dict=True,
        )
        if not np.isfinite(float(metrics["loss"])):
            raise RuntimeError(f"{architecture} produced a non-finite loss")
        print(
            f"{architecture}: parameters={model.count_params():,}, "
            f"loss={float(metrics['loss']):.4f}"
        )

    memory = tf.config.experimental.get_memory_info("GPU:0")
    if int(memory["peak"]) <= 0:
        raise RuntimeError("No TensorFlow GPU memory was used")
    print(f"GPU peak TensorFlow allocation: {memory['peak'] / 2**20:.1f} MiB")
    print("GPU MODEL CHECK: PASS")


if __name__ == "__main__":
    main()
