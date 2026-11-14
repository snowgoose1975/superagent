"""按固定计划训练表格分类模型，并生成验证集预测文件。

本模块只支持能力清单中公布的随机森林和逻辑回归。验证数据只用于
生成预测，不读取验证标签，也不计算任何评分。通常由 executor.py
在独立子进程中调用本文件。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from executor_c.contracts import (
    ContractError,
    load_prepared_data,
    read_json,
    validate_plan,
    validate_prediction_file,
    write_json,
)
from executor_c.model_factory import build_estimator
from executor_c.diagnostics import experiment_signature, make_diagnostics, write_worker_snapshot
from data_b.modality_readers import combine_text_columns


def _construct_model(plan: dict[str, Any], manifest: dict[str, Any]) -> Any:
    """根据已校验计划构造白名单内的模型，不接受动态类名或代码。"""
    try:
        return build_estimator(
            plan["model"],
            plan["params"],
            plan["random_seed"],
            representation=manifest.get("representation", "dense_numeric"),
            pipeline=plan.get("pipeline"),
            modality=manifest.get("modality"),
        )
    except ValueError as exc:
        raise ContractError("UNSUPPORTED_MODEL", "模型构造请求不受支持") from exc


def _ordered_predictions(
    model: Any, X_valid: np.ndarray, class_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """将模型概率列按统一类别 id 升序排列，并检查输出形状和数值。"""
    raw_pred = np.asarray(model.predict(X_valid))
    raw_prob = np.asarray(model.predict_proba(X_valid), dtype=np.float64)
    expected_shape = (len(X_valid), len(class_ids))
    if raw_pred.shape != (len(X_valid),) or raw_prob.shape != expected_shape:
        raise RuntimeError("模型预测的数组形状与验证样本数或类别数不符")
    model_classes = np.asarray(model.classes_)
    if model_classes.ndim != 1 or len(model_classes) != len(class_ids):
        raise RuntimeError("模型类别列表形状错误")
    try:
        column_for_id = {int(class_id): i for i, class_id in enumerate(model_classes.tolist())}
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("模型返回了无法识别的类别 id") from exc
    if len(column_for_id) != len(class_ids) or set(column_for_id) != set(class_ids):
        raise RuntimeError("模型训练所得类别与数据清单类别不一致")
    probs = raw_prob[:, [column_for_id[class_id] for class_id in class_ids]]
    if not np.isfinite(probs).all() or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise RuntimeError("模型概率包含非有限值或超出 [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("模型输出的每行概率和不为 1")
    if not np.isin(raw_pred, class_ids).all():
        raise RuntimeError("模型预测包含清单中不存在的类别 id")
    return raw_pred.astype(np.int64), probs


def _diagnostic_array_bytes(values: np.ndarray) -> int:
    if values.dtype.kind == "O":
        return int(values.nbytes + sum(len(item.encode("utf-8")) for item in values.flat))
    return int(values.nbytes)


def _write_predictions(
    path: Path,
    sample_ids: list[str],
    class_ids: list[int],
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """以 UTF-8 和协议规定的列顺序写预测 CSV，拒绝覆盖旧文件。"""
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample_id", "y_pred", *(f"p_{class_id}" for class_id in class_ids)])
        for sample_id, prediction, row in zip(sample_ids, predictions, probabilities, strict=True):
            writer.writerow([sample_id, int(prediction), *(format(float(p), ".17g") for p in row)])


def run(plan_path: Path, manifest_path: Path, run_dir: Path) -> None:
    """读取输入、训练、保存并复核产物；run_dir 必须由调用方预先建立。"""
    started = time.perf_counter()
    fitting_started = False
    phase = "input"
    experiment_id = run_dir.name
    random_seed: int | None = None
    data = None
    model = None
    fit_complete = False
    timings: dict[str, float | None] = {
        "input": None, "fit": None, "predict": None, "serialize": None,
        "reload_check": None, "total": None,
    }
    captured_warnings: list[dict[str, str]] = []
    warnings_truncated = False
    signature: str | None = None

    def snapshot(status: str, failure: dict[str, str] | None = None) -> None:
        timings["total"] = max(0.0, time.perf_counter() - started)
        try:
            write_worker_snapshot(
                run_dir / "worker_diagnostics.json",
                make_diagnostics(
                    experiment_id=experiment_id,
                    status=status,
                    phase="complete" if status == "success" else phase,
                    timings_seconds=timings,
                    model_bytes=(
                        (run_dir / "model.joblib").stat().st_size
                        if (run_dir / "model.joblib").is_file()
                        else None
                    ),
                    input_shape={
                        "train": list(data.X_train.shape) if data is not None else None,
                        "validation": list(data.X_valid.shape) if data is not None else None,
                    },
                    input_matrix_bytes=(
                        _diagnostic_array_bytes(data.X_train)
                        + _diagnostic_array_bytes(data.X_valid)
                        if data is not None
                        else None
                    ),
                    random_seed=random_seed,
                    effective_iterations=(
                        _effective_iterations(model) if fit_complete and model is not None else None
                    ),
                    warnings=captured_warnings,
                    warnings_truncated=warnings_truncated,
                    failure=failure,
                    signature=signature,
                ),
            )
        except Exception:
            # Diagnostic failure must not replace the original worker outcome.
            pass

    try:
        input_started = time.perf_counter()
        raw_plan = read_json(plan_path, error_code="PLAN_INVALID")
        plan = validate_plan(raw_plan)
        experiment_id = plan["experiment_id"]
        random_seed = plan["random_seed"]
        data = load_prepared_data(manifest_path, plan)
        manifest = data.manifest
        signature = experiment_signature(plan, data.input_hashes)
        timings["input"] = time.perf_counter() - input_started
        snapshot("running")
        model = _construct_model(plan, manifest)
        if manifest.get("representation") == "raw_text":
            modality = manifest["modality"]
            X_train = combine_text_columns(data.X_train, modality["text_columns"])
            X_valid = combine_text_columns(data.X_valid, modality["text_columns"])
        else:
            X_train, X_valid = data.X_train, data.X_valid
        print(f"模型：{plan['model']}", flush=True)

        # fit 开始后发生的错误属于执行失败，而不是输入拒绝。
        fitting_started = True
        phase = "fit"
        snapshot("running")
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, data.y_train)
        fit_complete = True
        timings["fit"] = time.perf_counter() - fit_started
        warnings_truncated = len(caught) > 20
        captured_warnings = [
            {
                "category": warning.category.__name__,
                "phase": "fit",
                "message": str(warning.message),
            }
            for warning in caught[:20]
        ]
        snapshot("running")
        print("训练完成。", flush=True)
        phase = "prediction"
        snapshot("running")
        predict_started = time.perf_counter()
        predictions, probabilities = _ordered_predictions(model, X_valid, data.class_ids)
        timings["predict"] = time.perf_counter() - predict_started

        phase = "serialize"
        snapshot("running")
        serialize_started = time.perf_counter()
        model_path = run_dir / "model.joblib"
        # joblib 写入已独占创建的文件，避免静默覆盖历史产物。
        with model_path.open("xb") as stream:
            joblib.dump(model, stream)
        timings["serialize"] = time.perf_counter() - serialize_started

        # 在同一 worker 进程重新加载并探测少量样本；跨进程加载由集成测试负责。
        phase = "reload_check"
        snapshot("running")
        reload_started = time.perf_counter()
        restored = joblib.load(model_path)
        probe_size = min(len(data.X_valid), 5)
        restored_pred, restored_prob = _ordered_predictions(
            restored, X_valid[:probe_size], data.class_ids
        )
        if not np.array_equal(restored_pred, predictions[:probe_size]) or not np.allclose(
            restored_prob, probabilities[:probe_size], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("保存后重新加载的模型预测与原模型不一致")
        timings["reload_check"] = time.perf_counter() - reload_started

        prediction_path = run_dir / "predictions.csv"
        _write_predictions(
            prediction_path, data.sample_ids, data.class_ids, predictions, probabilities
        )
        count = validate_prediction_file(
            prediction_path, sample_ids=data.sample_ids, class_ids=data.class_ids
        )
        write_json(
            run_dir / "worker_metadata.json",
            {
                "class_ids": data.class_ids,
                "n_predictions": count,
                "input_hashes": data.input_hashes,
            },
        )
        snapshot("success")
        print("预测与模型产物已保存并通过检查。", flush=True)
    except ContractError as exc:
        status = (
            "rejected"
            if exc.code
            in {"PLAN_INVALID", "DATA_INVALID", "UNSUPPORTED_MODEL", "UNSUPPORTED_PARAMETER"}
            and not fitting_started
            else "failed"
        )
        snapshot(status, {"code": exc.code, "message": exc.message})
        _record_error(run_dir, status, exc.code, exc.message)
        raise
    except Exception as exc:
        if phase in {"serialize", "reload_check"}:
            code = "ARTIFACT_INVALID"
        else:
            code = "TRAINING_ERROR" if fitting_started else "DATA_INVALID"
        snapshot("failed", {"code": code, "message": f"{type(exc).__name__}: {exc}"})
        _record_error(run_dir, "failed", code, str(exc))
        raise


def _effective_iterations(model: Any) -> int | None:
    if model is None:
        return None
    if hasattr(model, "steps") and model.steps:
        model = model.steps[-1][1]
    if hasattr(model, "n_estimators"):
        value = getattr(model, "n_estimators")
    elif hasattr(model, "n_iter_"):
        value = np.asarray(model.n_iter_).max()
    elif hasattr(model, "named_steps"):
        estimator = model.named_steps.get("logisticregression")
        value = (
            np.asarray(estimator.n_iter_).max()
            if estimator is not None and hasattr(estimator, "n_iter_")
            else None
        )
    else:
        value = None
    return int(value) if value is not None and np.isfinite(value) else None


def _record_error(run_dir: Path, status: str, code: str, message: str) -> None:
    """尽力写入机器可读错误；保留 stderr traceback 供开发排查。"""
    print(traceback.format_exc(), file=sys.stderr, end="")
    try:
        write_json(
            run_dir / "worker_error.json",
            {"status": status, "error": {"code": code, "message": message}},
        )
    except Exception:
        print("无法写入 worker_error.json；原始异常见上述 traceback。", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按固定计划训练白名单内的分类模型")
    parser.add_argument("--plan", required=True, type=Path, help="实验计划 JSON 的绝对路径")
    parser.add_argument(
        "--data-manifest", required=True, type=Path, help="B 数据清单 JSON 的绝对路径"
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="调用方已创建的本次运行目录")
    args = parser.parse_args(argv)
    try:
        run(args.plan, args.data_manifest, args.run_dir)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
