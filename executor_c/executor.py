"""Run the fixed training subprocess with a validated configuration."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from executor_c.contracts import (
    ContractError,
    InvocationError,
    environment_info,
    inspect_artifacts,
    publish_json,
    read_json,
    sha256_file,
    validate_identity,
    validate_manifest,
    validate_plan,
    write_json,
)
from executor_c.diagnostics import make_diagnostics, publish_final, read_worker_snapshot

WORKER_PATH = Path(__file__).resolve().with_name("train.py")
_SCHEMA_VERSION = "1.0"


def _error_parts(exc: BaseException, fallback: str) -> tuple[str, str]:
    code = getattr(exc, "code", fallback)
    message = getattr(exc, "message", str(exc) or fallback)
    return str(code), str(message)


def _write_terminal(run_dir: Path, result: dict[str, Any]) -> None:
    """Publish the terminal state atomically so readers see complete JSON."""
    publish_json(run_dir / "result.json", result)


def _terminate(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            # The fixed worker has no children. On Windows, terminate it directly and wait.
            proc.kill()
    except ProcessLookupError:
        pass
    finally:
        try:
            # A worker stuck in uninterruptible I/O cannot be reaped; never block forever.
            proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _base_result(task_id: str, data_version: str, experiment_id: str) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "task_id": task_id,
        "data_version": data_version,
        "experiment_id": experiment_id,
        "status": "failed",
        "duration_seconds": 0.0,
        "returncode": None,
        "n_predictions": None,
        "class_ids": None,
        "artifacts": {
            "model_path": None,
            "predictions_path": None,
            "stdout_path": "stdout.log",
            "stderr_path": "stderr.log",
        },
        "error": None,
    }


def execute(
    plan: dict[str, Any],
    data_manifest_path: Path | str,
    runs_root: Path | str,
    *,
    encoder_runtime: dict[str, Any] | None = None,
    no_time_limit: bool = False,
) -> str:
    """Run one experiment and return its result.json POSIX path relative to runs_root."""
    started = time.monotonic()
    root = Path(runs_root).expanduser()
    manifest_path = Path(data_manifest_path).expanduser()
    if not root.is_absolute() or not manifest_path.is_absolute():
        raise InvocationError("PATH_INVALID", "runs_root and data_manifest_path must be absolute paths")
    root = root.resolve()
    manifest_path = manifest_path.resolve()

    # Validate run identifiers before constructing any paths.
    task_id, data_version, experiment_id = validate_identity(plan)
    run_dir = root / task_id / data_version / experiment_id
    resolved_run_dir = run_dir.resolve()
    try:
        resolved_run_dir.relative_to(root)
    except ValueError as exc:
        raise InvocationError("PATH_INVALID", "The resolved experiment directory is outside runs_root") from exc
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise InvocationError("RUN_EXISTS", f"The experiment directory already exists: {run_dir}") from exc
    except OSError as exc:
        raise InvocationError("RUN_CREATE_FAILED", f"Cannot create the experiment directory: {exc}") from exc

    result = _base_result(task_id, data_version, experiment_id)
    relpath = (Path(task_id) / data_version / experiment_id / "result.json").as_posix()
    proc: subprocess.Popen[Any] | None = None
    stdout_handle = None
    stderr_handle = None
    publish_terminal = True
    try:
        # Create new log files for the experiment so the worker can append to them.
        (run_dir / "stdout.log").open("x", encoding="utf-8").close()
        (run_dir / "stderr.log").open("x", encoding="utf-8").close()

        # Snapshot the relevant sources for reproducibility and audit.
        source_paths = {
            "train.py": (WORKER_PATH, "train.snapshot.py"),
            "contracts.py": (Path(__file__).with_name("contracts.py"), "contracts.snapshot.py"),
            "executor.py": (Path(__file__).resolve(), "executor.snapshot.py"),
            "model_registry.py": (
                Path(__file__).with_name("model_registry.py"),
                "model_registry.snapshot.py",
            ),
            "model_factory.py": (
                Path(__file__).with_name("model_factory.py"),
                "model_factory.snapshot.py",
            ),
            "capabilities.json": (
                Path(__file__).with_name("capabilities.json"),
                "capabilities.snapshot.json",
            ),
            "diagnostics.py": (
                Path(__file__).with_name("diagnostics.py"),
                "diagnostics.snapshot.py",
            ),
            "modalities.py": (
                Path(__file__).with_name("modalities.py"),
                "modalities.snapshot.py",
            ),
            "modality_readers.py": (
                Path(__file__).resolve().parents[1] / "data_b" / "modality_readers.py",
                "modality_readers.snapshot.py",
            ),
            "npy_reader.py": (
                Path(__file__).resolve().parents[1] / "data_b" / "npy_reader.py",
                "npy_reader.snapshot.py",
            ),
        }
        if plan.get("model") == "catboost":
            source_paths["native_table.py"] = (
                Path(__file__).with_name("native_table.py"), "native_table.snapshot.py",
            )
        pipeline = plan.get("pipeline")
        if isinstance(pipeline, dict) and "text_encoder" in pipeline:
            source_paths["text_encoder.py"] = (
                Path(__file__).with_name("text_encoder.py"), "text_encoder.snapshot.py",
            )
        hashes: dict[str, str] = {}
        for name, (source_path, snapshot_name) in source_paths.items():
            if source_path.is_file():
                snapshot_path = run_dir / snapshot_name
                with (
                    source_path.open("rb") as source,
                    snapshot_path.open("xb") as snapshot,
                ):
                    snapshot.write(source.read())
                hashes[name] = sha256_file(snapshot_path)
        write_json(run_dir / "source_hashes.json", hashes)
        write_json(run_dir / "environment.json", environment_info())

        # Complete input validation before writing the request.
        try:
            normalized_plan = validate_plan(plan)
            manifest = validate_manifest(manifest_path, normalized_plan)
        except (ContractError, InvocationError, OSError, ValueError) as exc:
            code, message = _error_parts(exc, "DATA_INVALID")
            # Record only validated identifiers and the error for a rejected request.
            write_json(
                run_dir / "request.json",
                {
                    "task_id": task_id,
                    "data_version": data_version,
                    "experiment_id": experiment_id,
                    "rejected": True,
                },
            )
            result.update(status="rejected", error={"code": code, "message": message})
            raise _Rejected

        # Persist the validated plan and manifest.
        write_json(run_dir / "request.json", normalized_plan)
        write_json(run_dir / "plan.json", normalized_plan)
        if "text_encoder" in normalized_plan.get("pipeline", {}):
            write_json(run_dir / "encoder_runtime.json", encoder_runtime or {"enabled": False})
        write_json(run_dir / "data_manifest.snapshot.json", manifest)

        worker_env = os.environ.copy()
        project_root = Path(__file__).resolve().parents[1]
        existing_pythonpath = worker_env.get("PYTHONPATH")
        worker_env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(project_root), existing_pythonpath) if part
        )
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            worker_env[key] = "1"
        worker_env["PYTHONIOENCODING"] = "utf-8"
        stdout_handle = (run_dir / "stdout.log").open("a", encoding="utf-8", newline="")
        stderr_handle = (run_dir / "stderr.log").open("a", encoding="utf-8", newline="")
        kwargs: dict[str, Any] = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(WORKER_PATH),
                "--plan",
                str(run_dir / "plan.json"),
                "--data-manifest",
                str(manifest_path),
                "--run-dir",
                str(run_dir),
            ],
            cwd=str(run_dir),
            env=worker_env,
            shell=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
            **kwargs,
        )
        try:
            if no_time_limit or (
                "text_encoder" in normalized_plan.get("pipeline", {})
                and (encoder_runtime or {}).get("no_time_limit") is True
            ):
                print(f"[C] {experiment_id} started locally. Progress log: {run_dir / 'stdout.log'}",
                      flush=True)
                while True:
                    try:
                        returncode = proc.wait(timeout=45)
                        break
                    except subprocess.TimeoutExpired:
                        elapsed = round(time.monotonic() - started)
                        print(f"[C] {experiment_id} is still running after {elapsed} seconds.", flush=True)
            else:
                returncode = proc.wait(timeout=normalized_plan["timeout_seconds"])
        except subprocess.TimeoutExpired:
            _terminate(proc)
            result.update(
                status="timeout",
                returncode=proc.returncode,
                error={"code": "TIMEOUT", "message": "Training exceeded the plan's time limit and the subprocess was terminated"},
            )
            raise _Finished
        finally:
            stdout_handle.close()
            stderr_handle.close()
            stdout_handle = stderr_handle = None

        result["returncode"] = returncode
        worker_error_path = run_dir / "worker_error.json"
        if worker_error_path.is_file():
            try:
                worker_error = read_json(worker_error_path, "TRAINING_ERROR", max_bytes=1024 * 1024)
                status = worker_error.get("status")
                if status in ("rejected", "failed"):
                    error = worker_error.get("error")
                    if not isinstance(error, dict):
                        error = {}
                    result.update(
                        status=status,
                        error={
                            "code": str(
                                error.get("code")
                                or ("PLAN_INVALID" if status == "rejected" else "TRAINING_ERROR")
                            ),
                            "message": str(error.get("message") or "The training worker reported a failure"),
                        },
                    )
                    raise _Finished
            except (ContractError, ValueError):
                pass
        if returncode != 0:
            result.update(
                status="failed",
                error={"code": "TRAINING_ERROR", "message": f"The training worker exited with code {returncode}"},
            )
            raise _Finished

        try:
            inspected = inspect_artifacts(run_dir, manifest_path, manifest)
        except (ContractError, OSError, ValueError) as exc:
            code, message = _error_parts(exc, "ARTIFACT_INVALID")
            result.update(status="failed", error={"code": code, "message": message})
            raise _Finished
        result.update(
            status="success",
            n_predictions=inspected["n_predictions"],
            class_ids=inspected["class_ids"],
            artifacts={
                "model_path": "model.joblib",
                "predictions_path": "predictions.csv",
                "stdout_path": "stdout.log",
                "stderr_path": "stderr.log",
            },
            error=None,
        )
    except _Rejected:
        pass
    except _Finished:
        pass
    except KeyboardInterrupt:
        # Decide to suppress the terminal before cleanup: cleanup may itself be interrupted.
        publish_terminal = False
        if proc is not None:
            _terminate(proc)
        raise
    except Exception as exc:
        code, message = _error_parts(exc, "TRAINING_ERROR")
        result.update(status="failed", error={"code": code, "message": message})
    finally:
        # Stop the worker if waiting or recording the result fails.
        if proc is not None and proc.poll() is None:
            _terminate(proc)
        if proc is not None and result["returncode"] is None:
            result["returncode"] = proc.returncode
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
        result["duration_seconds"] = round(max(0.0, time.monotonic() - started), 6)
        try:
            worker_diagnostics = read_worker_snapshot(run_dir / "worker_diagnostics.json")
            if worker_diagnostics is None:
                worker_diagnostics = make_diagnostics(
                    experiment_id=experiment_id,
                    status=result["status"],
                    phase=(
                        "worker_timeout"
                        if result["status"] == "timeout"
                        else "input"
                        if result["status"] == "rejected"
                        else "execution"
                    ),
                )
            diagnostics = dict(worker_diagnostics)
            diagnostics["experiment_id"] = experiment_id
            diagnostics["status"] = result["status"]
            timings = dict(diagnostics.get("timings_seconds", {}))
            timings["total"] = result["duration_seconds"]
            diagnostics["timings_seconds"] = timings
            if diagnostics.get("phase") == "complete" and result["status"] != "success":
                diagnostics["phase"] = "artifact"
            if result.get("status") != "success":
                error = result.get("error") or {}
                diagnostics["failure"] = {
                    "code": str(error.get("code", "UNKNOWN"))[:100],
                    "message": str(error.get("message", ""))[:500].replace("\n", " "),
                }
            model_path = run_dir / "model.joblib"
            diagnostics["model_bytes"] = (
                model_path.stat().st_size if model_path.is_file() else None
            )
            publish_final(run_dir / "diagnostics.json", diagnostics)
        except Exception as diagnostic_error:
            print(
                f"Could not publish structured diagnostics: {type(diagnostic_error).__name__}",
                file=sys.stderr,
            )
        if publish_terminal:
            _write_terminal(run_dir, result)
    return relpath


class _Rejected(Exception):
    """Signal a rejected terminal state after invalid input."""


class _Finished(Exception):
    """Signal a terminal state and skip further training."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one executor experiment")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--runs-root", required=True)
    args = parser.parse_args(argv)
    try:
        plan = read_json(Path(args.plan), "PLAN_INVALID")
        result_relpath = execute(plan, args.data_manifest, args.runs_root)
        print(result_relpath)
        result_path = Path(args.runs_root).resolve() / Path(*result_relpath.split("/"))
        result_obj = read_json(result_path)
        return 0 if result_obj.get("status") == "success" else 1
    except (ContractError, InvocationError, OSError, ValueError) as exc:
        _, message = _error_parts(exc, "INVOCATION_ERROR")
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
