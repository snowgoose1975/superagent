"""真实训练和边界行为验证；全部数据由本文件模拟，不使用真实任务测试集。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from executor_c import executor
from executor_c.consumer import load_successful_predictions
from executor_c.contracts import (
    ContractError,
    InvocationError,
    capabilities,
    load_prepared_data,
    read_json,
    resolve_data_path,
    validate_plan,
    validate_prediction_file,
)
from executor_c.make_demo import make_demo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "executor_c"


def plan_for(manifest_path: Path, model: str = "random_forest", experiment: str = "exp_001"):
    manifest = read_json(manifest_path)
    return {
        "schema_version": "1.0",
        "task_id": manifest["task_id"],
        "data_version": manifest["data_version"],
        "experiment_id": experiment,
        "model": model,
        "params": {"n_estimators": 12, "max_depth": 4} if model == "random_forest" else {},
        "random_seed": 42,
        "timeout_seconds": 60,
    }


def create_data(tmp_path: Path, n_classes: int = 2) -> Path:
    return make_demo(tmp_path / "B data", task_id="task_demo", n_classes=n_classes)


def read_result(root: Path, relative: str):
    path = root / relative
    return path, read_json(path)


def rewrite_json(path: Path, value: dict):
    # 仅修改本测试新建的模拟输入，用于验证拒绝行为；不修改历史运行。
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.mark.parametrize(
    "model",
    ["random_forest", "logistic_regression", "extra_trees", "hist_gradient_boosting"],
)
@pytest.mark.parametrize("n_classes", [2, 3])
def test_real_training_reload_and_d_handoff(tmp_path, model, n_classes):
    manifest_path = create_data(tmp_path, n_classes)
    root = (tmp_path / "shared runs with spaces").resolve()
    plan = plan_for(manifest_path, model)
    result_ref = executor.execute(plan, manifest_path, root)
    result_path, result = read_result(root, result_ref)
    assert result["status"] == "success", result
    assert result["returncode"] == 0
    assert result["class_ids"] == list(range(n_classes))
    assert result["error"] is None
    assert result["duration_seconds"] > 0
    assert result_ref == "task_demo/v1/exp_001/result.json"
    assert "metrics" not in result
    assert read_json(result_path.parent / "plan.json")["params"]
    assert (result_path.parent / "train.snapshot.py").is_file()
    assert (result_path.parent / "contracts.snapshot.py").is_file()
    assert (result_path.parent / "executor.snapshot.py").is_file()
    assert (result_path.parent / "model_registry.snapshot.py").is_file()
    assert (result_path.parent / "model_factory.snapshot.py").is_file()
    assert (result_path.parent / "diagnostics.snapshot.py").is_file()
    diagnostics = read_json(result_path.parent / "diagnostics.json")
    assert diagnostics["status"] == "success"
    assert diagnostics["experiment_id"] == plan["experiment_id"]
    assert diagnostics["random_seed"] == plan["random_seed"]
    assert diagnostics["input_shape"]["validation"][0] == 30
    assert diagnostics["timings_seconds"]["total"] >= diagnostics["timings_seconds"]["fit"]
    assert "训练完成" in (result_path.parent / "stdout.log").read_text(encoding="utf-8")

    returned_result, rows = load_successful_predictions(result_ref, root)
    assert returned_result == result
    assert len(rows) == result["n_predictions"] == 30
    assert list(rows[0]) == ["sample_id", "y_pred", *[f"p_{i}" for i in range(n_classes)]]
    assert rows[0]["sample_id"] == "valid_000001"

    # 用全新 Python 进程导入 sklearn 并加载，避免“仅原内存中的模型可用”。
    script = (
        "import json,sys,joblib,numpy as np; "
        "m=joblib.load(sys.argv[1]); X=np.load(sys.argv[2],allow_pickle=False); "
        "print(json.dumps({'pred':m.predict(X).tolist(),'proba':m.predict_proba(X).tolist()}))"
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(result_path.parent / "model.joblib"),
            str(manifest_path.parent / "X_valid.npy"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    fresh = json.loads(process.stdout)
    assert fresh["pred"] == [int(row["y_pred"]) for row in rows]
    np.testing.assert_allclose(
        fresh["proba"],
        [[float(row[f"p_{i}"]) for i in range(n_classes)] for row in rows],
        atol=1e-12,
        rtol=0,
    )
    if model == "logistic_regression":
        saved = joblib.load(result_path.parent / "model.joblib")
        X_train = np.load(manifest_path.parent / "X_train.npy", allow_pickle=False)
        np.testing.assert_allclose(saved.steps[0][1].mean_, X_train.mean(axis=0))


@pytest.mark.parametrize("case", ["unknown_model", "unknown_parameter", "bool", "range", "version"])
def test_plan_rejection_is_recorded_without_training(tmp_path, case):
    manifest_path = create_data(tmp_path)
    plan = plan_for(manifest_path)
    if case == "unknown_model":
        plan["model"] = "anything_else"
    elif case == "unknown_parameter":
        plan["params"]["max_iteration"] = 100
    elif case == "bool":
        plan["params"]["n_estimators"] = True
    elif case == "range":
        plan["timeout_seconds"] = 601
    else:
        plan["schema_version"] = "2.0"
    root = (tmp_path / "runs").resolve()
    ref = executor.execute(plan, manifest_path, root)
    result_path, result = read_result(root, ref)
    assert result["status"] == "rejected"
    assert result["returncode"] is None
    assert result["artifacts"]["predictions_path"] is None
    assert not (result_path.parent / "model.joblib").exists()
    with pytest.raises(ValueError):
        load_successful_predictions(ref, root)


def test_unexpected_fields_are_not_saved_as_raw_input(tmp_path):
    manifest_path = create_data(tmp_path)
    plan = plan_for(manifest_path)
    plan["unexpected"] = "test-private-value-never-record"
    root = (tmp_path / "runs").resolve()
    ref = executor.execute(plan, manifest_path, root)
    run_dir = (root / ref).parent
    assert read_json(root / ref)["status"] == "rejected"
    for name in ("request.json", "result.json", "stdout.log", "stderr.log"):
        assert "test-private-value-never-record" not in (run_dir / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "case",
    [
        "nan",
        "shape",
        "label_shape",
        "missing_class",
        "duplicate_ids",
        "missing_file",
        "version",
        "object",
    ],
)
def test_bad_b_data_is_rejected(tmp_path, case):
    manifest_path = create_data(tmp_path)
    base = manifest_path.parent
    plan = plan_for(manifest_path)
    manifest = read_json(manifest_path)
    if case in ("nan", "shape", "object"):
        values = np.load(base / "X_train.npy", allow_pickle=False)
        if case == "nan":
            values[0, 0] = np.nan
        elif case == "shape":
            values = values[:, :-1]
        else:
            values = values.astype(object)
        with (base / "X_train.npy").open("wb") as stream:
            np.save(stream, values, allow_pickle=case == "object")
    elif case in ("label_shape", "missing_class"):
        labels = np.load(base / "y_train.npy", allow_pickle=False)
        labels = labels[:, None] if case == "label_shape" else np.zeros_like(labels)
        np.save(base / "y_train.npy", labels, allow_pickle=False)
    elif case == "duplicate_ids":
        ids = ["sample_id", *(["same_id"] * manifest["validation"]["n_samples"])]
        (base / "valid_ids.csv").write_text("\n".join(ids) + "\n", encoding="utf-8")
    elif case == "missing_file":
        manifest["train"]["features_path"] = "absent.npy"
        rewrite_json(manifest_path, manifest)
    else:
        manifest["data_version"] = "v2"
        rewrite_json(manifest_path, manifest)
    root = (tmp_path / "runs").resolve()
    ref = executor.execute(plan, manifest_path, root)
    _, result = read_result(root, ref)
    assert result["status"] == "rejected", result
    assert result["error"]["code"] == "DATA_INVALID", result
    assert result["artifacts"]["model_path"] is None


def test_training_inputs_do_not_read_evaluation_labels(tmp_path, monkeypatch):
    manifest_path = create_data(tmp_path)
    original_open = Path.open
    opened = []

    def audited_open(path, *args, **kwargs):
        opened.append(path.name)
        assert path.name not in ("validation_labels.csv", "evaluation_manifest.json")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", audited_open)
    data = load_prepared_data(manifest_path, validate_plan(plan_for(manifest_path)))
    assert data.X_valid.shape[0] == 30
    assert not any("evaluation" in key or "valid_labels" in key for key in data.input_hashes)
    assert "valid_ids.csv" in opened


@pytest.mark.parametrize(
    "relative", ["../outside.npy", "/etc/passwd", "C:/outside.npy", "sub\\file.npy"]
)
def test_data_paths_cannot_escape_package(tmp_path, relative):
    with pytest.raises(ContractError) as exc:
        resolve_data_path(tmp_path, relative)
    assert exc.value.code == "DATA_INVALID"


def test_duplicate_experiment_never_overwrites_result(tmp_path):
    manifest_path = create_data(tmp_path)
    plan = plan_for(manifest_path)
    root = (tmp_path / "runs").resolve()
    ref = executor.execute(plan, manifest_path, root)
    previous = (root / ref).read_bytes()
    with pytest.raises(InvocationError) as exc:
        executor.execute(plan, manifest_path, root)
    assert exc.value.code == "RUN_EXISTS"
    assert (root / ref).read_bytes() == previous


def test_invalid_identity_and_relative_roots_fail_before_creation(tmp_path):
    manifest_path = create_data(tmp_path)
    plan = plan_for(manifest_path)
    plan["experiment_id"] = "../escape"
    with pytest.raises(InvocationError):
        executor.execute(plan, manifest_path, (tmp_path / "runs").resolve())
    with pytest.raises(InvocationError):
        executor.execute(plan_for(manifest_path), manifest_path, "relative_runs")


@pytest.mark.parametrize("case", ["exception", "empty_success", "worker_rejection", "timeout"])
def test_worker_failures_and_real_timeout(tmp_path, monkeypatch, case):
    manifest_path = create_data(tmp_path)
    worker = tmp_path / "controlled_worker.py"
    if case == "exception":
        script = "raise RuntimeError('controlled training failure')\n"
    elif case == "empty_success":
        script = "print('no artifacts were saved')\n"
    elif case == "worker_rejection":
        script = (
            "import json\nfrom pathlib import Path\n"
            "Path('worker_error.json').write_text(json.dumps({'status':'rejected','error':"
            "{'code':'DATA_INVALID','message':'controlled data rejection'}}),encoding='utf-8')\n"
            "raise SystemExit(1)\n"
        )
    else:
        script = "import time\nprint('sleeping',flush=True)\ntime.sleep(30)\n"
    worker.write_text(script, encoding="utf-8")
    monkeypatch.setattr(executor, "WORKER_PATH", worker)
    plan = plan_for(manifest_path)
    plan["timeout_seconds"] = 1 if case == "timeout" else 60
    root = (tmp_path / "runs").resolve()
    ref = executor.execute(plan, manifest_path, root)
    result_path, result = read_result(root, ref)
    expected_status = {
        "exception": "failed",
        "empty_success": "failed",
        "worker_rejection": "rejected",
        "timeout": "timeout",
    }[case]
    expected_code = {
        "exception": "TRAINING_ERROR",
        "empty_success": "ARTIFACT_INVALID",
        "worker_rejection": "DATA_INVALID",
        "timeout": "TIMEOUT",
    }[case]
    assert result["status"] == expected_status, result
    assert result["error"]["code"] == expected_code, result
    assert result["returncode"] is not None
    assert result["artifacts"]["model_path"] is None
    assert result["artifacts"]["predictions_path"] is None
    assert result["n_predictions"] is None
    if case == "timeout":
        assert result["returncode"] != 0
        assert result["duration_seconds"] < 15
        assert "sleeping" in (result_path.parent / "stdout.log").read_text(encoding="utf-8")
        diagnostics = read_json(result_path.parent / "diagnostics.json")
        assert diagnostics["status"] == "timeout"
        assert diagnostics["failure"]["code"] == "TIMEOUT"


def test_keyboard_interrupt_terminates_and_does_not_publish_terminal(tmp_path, monkeypatch):
    manifest_path = create_data(tmp_path)

    class FakeProcess:
        pid = 123456789
        returncode = None
        terminated = False

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if timeout is not None:
                raise KeyboardInterrupt
            self.returncode = -1
            return self.returncode

        def kill(self):
            self.terminated = True
            self.returncode = -1

    fake = FakeProcess()
    monkeypatch.setattr(executor.subprocess, "Popen", lambda *args, **kwargs: fake)
    if os.name != "nt":
        monkeypatch.setattr(executor.os, "killpg", lambda pid, sig: fake.kill())
    root = (tmp_path / "runs").resolve()
    with pytest.raises(KeyboardInterrupt):
        executor.execute(plan_for(manifest_path), manifest_path, root)
    assert fake.terminated
    assert not (root / "task_demo/v1/exp_001/result.json").exists()


def test_wait_error_also_terminates_child(tmp_path, monkeypatch):
    manifest_path = create_data(tmp_path)

    class FakeProcess:
        pid = 123456789
        returncode = None
        terminated = False

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if timeout is not None:
                raise OSError("controlled wait failure")
            self.returncode = -1
            return -1

        def kill(self):
            self.terminated = True
            self.returncode = -1

    fake = FakeProcess()
    monkeypatch.setattr(executor.subprocess, "Popen", lambda *args, **kwargs: fake)
    if os.name != "nt":
        monkeypatch.setattr(executor.os, "killpg", lambda pid, sig: fake.kill())
    root = (tmp_path / "runs").resolve()
    ref = executor.execute(plan_for(manifest_path), manifest_path, root)
    _, result = read_result(root, ref)
    assert fake.terminated
    assert result["status"] == "failed"
    assert result["returncode"] == -1


@pytest.mark.parametrize("target", ["X_train.npy", "y_train.npy", "X_valid.npy", "manifest"])
def test_changed_inputs_are_not_published_as_success(tmp_path, monkeypatch, target):
    manifest_path = create_data(tmp_path)
    original = executor.inspect_artifacts

    def inspect_after_change(run_dir, path, manifest):
        if target == "manifest":
            changed = read_json(path)
            changed["feature_names"][0] = "modified_feature_name"
            rewrite_json(path, changed)
        else:
            file_path = manifest_path.parent / target
            data = np.load(file_path, allow_pickle=False)
            if target == "y_train.npy":
                data[0] = 1 - data[0]
            else:
                data[0, 0] += 100.0
            np.save(file_path, data, allow_pickle=False)
        return original(run_dir, path, manifest)

    monkeypatch.setattr(executor, "inspect_artifacts", inspect_after_change)
    root = (tmp_path / "runs").resolve()
    ref = executor.execute(plan_for(manifest_path), manifest_path, root)
    _, result = read_result(root, ref)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "ARTIFACT_INVALID"
    assert result["artifacts"]["predictions_path"] is None


@pytest.mark.parametrize("content", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', "[]"])
def test_nonstandard_or_duplicate_json_is_rejected(tmp_path, content):
    source = tmp_path / "invalid.json"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ContractError):
        read_json(source)


def test_probability_and_order_contract(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("sample_id,y_pred,p_0,p_1\nb,0,0.8,0.2\na,1,0.1,0.9\n", encoding="utf-8")
    with pytest.raises(ContractError):
        validate_prediction_file(path, sample_ids=["a", "b"], class_ids=[0, 1])
    path.write_text("sample_id,y_pred,p_0,p_1\na,0,nan,0.2\n", encoding="utf-8")
    with pytest.raises(ContractError):
        validate_prediction_file(path, sample_ids=["a"], class_ids=[0, 1])


def test_cli_works_from_another_working_directory(tmp_path):
    manifest_path = create_data(tmp_path)
    plan = plan_for(manifest_path)
    source = tmp_path / "plan.json"
    rewrite_json(source, plan)
    root = (tmp_path / "runs").resolve()
    process = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "executor.py"),
            "--plan",
            str(source),
            "--data-manifest",
            str(manifest_path),
            "--runs-root",
            str(root),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                part for part in (str(PROJECT_ROOT), os.environ.get("PYTHONPATH", "")) if part
            ),
        },
    )
    assert process.returncode == 0, process.stderr
    ref = process.stdout.strip()
    result, rows = load_successful_predictions(ref, root)
    assert result["status"] == "success" and len(rows) == 30


def test_demo_does_not_overwrite_existing_dataset(tmp_path):
    manifest = create_data(tmp_path)
    previous = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        create_data(tmp_path)
    assert manifest.read_bytes() == previous


def test_capability_file_matches_runtime_validator():
    assert read_json(PACKAGE_ROOT / "capabilities.json") == capabilities()


def test_consumer_rejects_extra_csv_fields(tmp_path):
    run_dir = tmp_path / "task_demo/v1/exp_001"
    run_dir.mkdir(parents=True)
    rewrite_json(
        run_dir / "result.json",
        {
            "schema_version": "1.0",
            "task_id": "task_demo",
            "data_version": "v1",
            "experiment_id": "exp_001",
            "status": "success",
            "class_ids": [0, 1],
            "n_predictions": 1,
            "artifacts": {"predictions_path": "predictions.csv"},
        },
    )
    (run_dir / "predictions.csv").write_text(
        "sample_id,y_pred,p_0,p_1\na,0,0.7,0.3,unexpected\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_successful_predictions("task_demo/v1/exp_001/result.json", tmp_path)
