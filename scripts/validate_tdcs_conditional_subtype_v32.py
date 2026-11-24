"""Audit S13's native predictions and the unaltered M22 crop stream; no fitting."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

import tdcs_raw_conditional_subtype_v32 as conditional
import validate_tdcs_raw_transfer_v28 as validator


ROOT = Path(__file__).resolve().parents[1]
raw = conditional.raw


def count_usage(trace, fitting, cfg):
    records = {r["id"]: r for r in fitting}
    counts = np.zeros(4, np.int64)
    eligible = 0
    for first in range(0, len(trace), cfg.micro_batch):
        has_fog = False
        for identifier, subject, center, stretch in trace[first:first+cfg.micro_batch]:
            r = records[identifier]
            assert r["subject"] == subject
            width = cfg.train_length
            coords = center + np.linspace(-width*stretch/2, width*stretch/2, width)
            n = len(r["signal"])
            near = np.clip(np.rint(coords).astype(int), 0, n-1)
            active = (coords >= 0) & (coords <= n-1) & r["valid"][near]
            labels = r["classes"][near][active]
            counts += np.bincount(labels, minlength=4)
            has_fog |= bool(np.any(labels > 0))
        eligible += int(has_fog)
    return {
        "valid_sampled_rows_by_class": counts.tolist(),
        "micros_with_any_FOG": eligible,
        "total_microbatches": len(trace) // cfg.micro_batch,
    }


def audit_matched(run, control, records, cfg):
    run_config = validator.read_json(run / "configuration.json")
    control_config = validator.read_json(control / "configuration.json")
    assert control_config["id"] == "M22" and control_config["source_sha256"] == conditional.weak.effective_source_hash()
    assert control_config["configuration"] == run_config["configuration"] == asdict(cfg)
    assert control_config["dataset_signature"] == run_config["dataset_signature"]
    assert validator.read_json(control / "auxiliary_pretraining_trace.json") == validator.read_json(run / "auxiliary_pretraining_trace.json")
    usage = validator.read_json(run / "conditional_usage.json")
    assert usage["source_sha256"] == conditional.effective_source_hash()
    assert usage["id"] == "S13" and usage["specification"] == conditional.SPEC
    assert not usage["outer_labels_used_for_training"] and usage["smoke_only"] == cfg.smoke
    assert len(usage["folds"]) == 5
    for fold, declared in enumerate(usage["folds"]):
        path = Path(f"fold_{fold}") / "training_trace.json"
        trace = validator.read_json(run / path)["fine_tune"]
        assert trace == validator.read_json(control / path)["fine_tune"]
        expected = count_usage(trace, [r for r in records if r["fold"] != fold], cfg)
        assert declared == {"fold": fold, "seed": cfg.seed+1009*fold+200000, **expected}
    return usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-output", type=Path, required=True)
    parser.add_argument("--weak-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--replay-progress", type=Path)
    args = parser.parse_args()
    for p in (args.result, args.replay_progress):
        assert p is None or p.resolve().is_relative_to(ROOT / "results")
    assert args.replay_progress is None or args.result.resolve() != args.replay_progress.resolve()
    reference = validator.read_json(ROOT / "results" / "tdcs_v29_M22_replay.json")
    assert reference["id"] == "M22" and reference["saved_model_replay"] and not reference["smoke_only"]
    control = (ROOT / reference["run"]).resolve()
    assert control.is_relative_to(ROOT / "kaggle-output")
    context = raw.data.load_context()
    with conditional.applied():
        audit = validator.audit(args.suite_output, conditional.METHOD, args.replay, context=context,
                                progress=args.replay_progress, weak_root=args.weak_root)
    run = Path(audit["run"])
    cfg = raw.Config(**validator.read_json(run / "configuration.json")["configuration"])
    records, _, _, _ = validator.load_records(context, False)
    audit["conditional_usage"] = audit_matched(run, control, records, cfg)
    audit["warmup_and_all_fine_tune_crops_match_M22"] = True
    raw.base.write(args.result, audit)
    print("S13 audited metrics:", audit["metrics"], flush=True)


if __name__ == "__main__":
    main()
