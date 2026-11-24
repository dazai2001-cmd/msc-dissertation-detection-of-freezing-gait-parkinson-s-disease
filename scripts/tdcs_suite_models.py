"""Bounded tree backends for suite experiments; CPU prediction is explicit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

import tdcs_suite_data as data


def fit_tree(kind, x, y, weights, seed, cfg, gpu, policy="SymmetricTree", groups=None):
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("Requested Kaggle CUDA backend is unavailable")
    if len(np.unique(y)) < 2:
        return {"constant": float(np.mean(y))}, {"backend": "constant", "actual_rounds": 0}
    if kind == "catboost":
        from catboost import CatBoostClassifier
        params = {"iterations": cfg.tree_rounds, "depth": 6, "learning_rate": .04,
            "l2_leaf_reg": 100., "loss_function": "Logloss", "boosting_type": "Plain",
            "bootstrap_type": "Bernoulli", "subsample": .85, "border_count": 128,
            "random_strength": 1., "thread_count": 2, "allow_writing_files": False,
            "grow_policy": policy, "random_seed": seed, "task_type": "GPU" if gpu else "CPU"}
        if policy != "SymmetricTree":
            params.update(score_function="L2", min_data_in_leaf=20)
        if policy == "Lossguide":
            params["max_leaves"] = 64
        if gpu:
            params.update(devices="0", gpu_ram_part=.55)
        model = CatBoostClassifier(**params).fit(x, y, sample_weight=weights, verbose=False)
        actual = model.get_all_params()
        assert actual["task_type"] == ("GPU" if gpu else "CPU")
        return model, {"backend": actual["task_type"], "actual_rounds": model.tree_count_,
            "configured": params, "resolved": actual}
    if kind in ("xgboost", "rank_map"):
        import xgboost as xgb
        params = {"objective": "rank:map" if kind == "rank_map" else "binary:logistic",
            "tree_method": "hist", "device": "cuda:0" if gpu else "cpu", "max_depth": 6,
            "eta": .035, "min_child_weight": 30 if not cfg.smoke else 1, "lambda": 30,
            "subsample": .85, "colsample_bytree": .85, "max_bin": 128,
            "seed": seed, "nthread": 2, "base_score": .5, "verbosity": 0}
        if kind == "rank_map":
            params.update(lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=4)
        matrix = xgb.DMatrix(x, label=y, nthread=2)
        if kind == "rank_map":
            if groups is None or int(np.sum(groups)) != len(y) or len(weights) != len(groups):
                raise ValueError("Ranking requires group counts and one weight per group")
            matrix.set_group(groups)
        matrix.set_weight(weights)
        model = xgb.train(params, matrix, num_boost_round=cfg.tree_rounds, verbose_eval=False)
        actual = json.loads(model.save_config())
        if gpu and not actual["learner"]["generic_param"]["device"].startswith("cuda"):
            raise RuntimeError("XGBoost silently fell back from CUDA")
        return model, {"backend": actual["learner"]["generic_param"]["device"],
            "actual_rounds": model.num_boosted_rounds(), "configured": params, "resolved": actual}
    if kind == "lightgbm":
        import lightgbm as lgb
        # Intentionally CPU on Kaggle, not an undisclosed GPU fallback.
        params = {"objective": "binary", "metric": "binary_logloss", "device_type": "cpu",
            "learning_rate": .04, "num_leaves": 31, "max_depth": 6,
            "min_data_in_leaf": 5 if cfg.smoke else 50, "min_data_in_bin": 1,
            "lambda_l2": 30., "feature_fraction": .85, "bagging_fraction": .85,
            "bagging_freq": 1, "max_bin": 127, "num_threads": 2, "seed": seed,
            "deterministic": True, "force_col_wise": True, "verbosity": -1}
        model = lgb.train(params, lgb.Dataset(x, label=y, weight=weights), num_boost_round=cfg.tree_rounds)
        return model, {"backend": "CPU_explicit_LightGBM", "actual_rounds": model.current_iteration(),
            "configured": params, "resolved": model.params}
    raise ValueError("Unknown declared learner")


def predict_tree(model, kind, x):
    if isinstance(model, dict) and "constant" in model:
        return np.full(len(x), model["constant"], np.float32)
    if kind == "catboost":
        p = model.predict_proba(x, task_type="CPU", thread_count=2)[:, 1]
    elif kind in ("xgboost", "rank_map"):
        import xgboost as xgb
        model.set_param({"device": "cpu", "nthread": 2})
        p = model.predict(xgb.DMatrix(x, nthread=2))
        if kind == "rank_map":
            p = expit(np.clip(p, -40, 40))
    elif kind == "lightgbm":
        p = model.predict(x, num_threads=2)
    else:
        raise ValueError("Unknown model kind")
    p = np.asarray(p, np.float32)
    if p.shape != (len(x),) or not np.isfinite(p).all():
        raise ValueError("Invalid saved-model predictions")
    return np.clip(p, 0, 1)


def save_tree(model, kind, stem):
    stem = Path(stem)
    if isinstance(model, dict):
        path = stem.with_suffix(".json")
        data.frozen.shared.common.write_json(path, model)
    elif kind == "catboost":
        path = stem.with_suffix(".cbm")
        model.save_model(str(path))
    elif kind in ("xgboost", "rank_map"):
        path = stem.with_suffix(".ubj")
        path.write_bytes(model.save_raw(raw_format="ubj"))
    else:
        path = stem.with_suffix(".txt")
        model.save_model(str(path))
    return path


def load_tree(kind, path):
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if kind == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier().load_model(str(path))
    if kind in ("xgboost", "rank_map"):
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(bytearray(path.read_bytes()))
        return model
    if kind == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_str=path.read_text())
    raise ValueError("Unknown model kind")


def ranking_plan(chosen, row_weights, records):
    order = np.argsort(records[chosen], kind="stable")
    selected, weights = chosen[order], row_weights[order]
    _, start, counts = np.unique(records[selected], return_index=True, return_counts=True)
    group_weights = np.asarray([weights[s:s+n].mean() for s, n in zip(start, counts)], np.float64)
    group_weights /= group_weights.mean()
    return selected, counts.astype(np.uint32), group_weights
