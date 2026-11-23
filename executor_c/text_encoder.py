"""Optional local Qwen encoder. Only the pinned public snapshot is downloaded."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .model_registry import TEXT_ENCODER_ID as ENCODER_ID
else:
    from model_registry import TEXT_ENCODER_ID as ENCODER_ID
REPO = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "ec126d058217e4e4af3695967f270934586127ef"
WEIGHT_BYTES = 1_191_586_416
WEIGHT_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
FILES = (
    "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "model.safetensors",
)
MODES = ("disabled", "cache-only", "download")
RUNTIME_PACKAGES = ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "filelock")
MAX_LENGTH = 256
DIMENSION = 512
DOWNLOAD_TIMEOUT = 1800
RUNTIME_TIMEOUT = 1800
MIN_FREE_BYTES = 2_000_000_000
MAX_FEATURE_BYTES = 256 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = Path(os.environ.get(
    "SUPERAGENT_ENCODER_CACHE", str(PROJECT_ROOT / ".cache" / "encoders")
)).expanduser().resolve()
RUNTIME = WEIGHTS_ROOT / f"runtime-cpu-v1-{sys.implementation.cache_tag}"


class EncoderError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def encoder_manifest() -> dict[str, Any]:
    return {
        "name": ENCODER_ID, "repository": REPO, "revision": REVISION,
        "weights_sha256": WEIGHT_SHA256, "max_length": MAX_LENGTH,
        "dimension": DIMENSION, "pooling": "last_token_l2", "adapter_version": 1,
    }


def dependency_versions() -> dict[str, str | None]:
    if (RUNTIME / ".ready").is_file() and str(RUNTIME) not in sys.path:
        sys.path.append(str(RUNTIME))
    versions = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _snapshot(root: Path) -> Path:
    return root / REVISION


def _complete(path: Path) -> bool:
    return (path / "snapshot.json").is_file() and all(
        (path / filename).is_file() for filename in FILES
    )


def encoder_status(mode: str, *, n_rows: int, n_fields: int) -> str:
    if mode == "disabled":
        return "disabled"
    dependencies_ready = all(dependency_versions().values())
    if not dependencies_ready and (mode == "cache-only" or not shutil.which("uv")):
        return "missing_dependencies"
    if n_rows * n_fields * DIMENSION * 4 > MAX_FEATURE_BYTES:
        return "feature_limit"
    if dependencies_ready and _complete(_snapshot(WEIGHTS_ROOT)):
        return "cached"
    if mode == "cache-only":
        return "cache_missing"
    cache_parent = next((p for p in (WEIGHTS_ROOT, *WEIGHTS_ROOT.parents) if p.is_dir()), PROJECT_ROOT)
    if shutil.disk_usage(cache_parent).free < MIN_FREE_BYTES:
        return "disk_limit"
    return "downloadable"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify(path: Path) -> None:
    if not _complete(path):
        raise EncoderError("ENCODER_CACHE_MISSING", "The local encoder weights cache is incomplete")
    try:
        manifest = json.loads((path / "snapshot.json").read_text(encoding="utf-8"))
        if (
            manifest["revision"] != REVISION
            or manifest["sha256"]["model.safetensors"] != WEIGHT_SHA256
            or any(_sha(path / name) != manifest["sha256"][name] for name in FILES)
        ):
            raise ValueError("checksum mismatch")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise EncoderError("ENCODER_CACHE_INVALID", "The encoder weights or configuration checksum does not match") from exc


def _download(root: Path) -> None:
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    target = _snapshot(root)
    from filelock import FileLock

    with FileLock(str(root / (REVISION + ".lock"))):
        if _complete(target):
            _verify(target)
            return
        from requests.exceptions import RequestException

        for attempt in range(3):
            try:
                _download_snapshot(root, target, snapshot_download)
                return
            except (OSError, RequestException, RuntimeError) as exc:
                if attempt == 2:
                    raise
                print(
                    f"[C] Weight transfer interrupted ({type(exc).__name__}). Resuming from the cache.",
                    flush=True,
                )
                time.sleep(2 ** (attempt + 1))


def _download_snapshot(root: Path, target: Path, snapshot_download: Any) -> None:
    source = Path(snapshot_download(
        repo_id=REPO, revision=REVISION, token=False, endpoint="https://huggingface.co",
        allow_patterns=list(FILES), cache_dir=root / "hub", max_workers=2,
    ))
    staging = Path(tempfile.mkdtemp(prefix=".download-", dir=root))
    for filename in FILES:
        # The staged snapshot lives on the same filesystem, so hard links avoid
        # storing a second copy of the model on filesystems that support them.
        source_file = (source / filename).resolve(strict=True)
        try:
            os.link(source_file, staging / filename)
        except OSError:
            shutil.copyfile(source_file, staging / filename)
    hashes = {name: _sha(staging / name) for name in FILES}
    if hashes["model.safetensors"] != WEIGHT_SHA256:
        raise EncoderError("ENCODER_CACHE_INVALID", "The encoder weights checksum does not match")
    (staging / "snapshot.json").write_text(
        json.dumps({"revision": REVISION, "sha256": hashes}), encoding="utf-8"
    )
    staging.rename(target)


def _public_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key.upper() not in {
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "SCHOOL_API_KEY",
    }}


def _prepare_runtime() -> bool:
    if all(dependency_versions().values()):
        return False
    uv = shutil.which("uv")
    if uv is None:
        raise EncoderError("ENCODER_DEPENDENCIES", "Automatic CPU encoder setup requires uv")
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".runtime-", dir=WEIGHTS_ROOT))
    print("[C] Preparing CPU dependencies for the selected encoder.", flush=True)
    try:
        result = subprocess.run(
            [uv, "--no-config", "pip", "install", "--torch-backend", "cpu",
             "--python", sys.executable, "--target", str(staging),
             "-r", str(PROJECT_ROOT / "requirements-encoder.txt")],
            env=_public_environment(), timeout=RUNTIME_TIMEOUT, check=False,
        )
        if result.returncode != 0:
            raise EncoderError("ENCODER_DEPENDENCIES", "Automatic installation of CPU encoder dependencies failed")
        installed = {
            distribution.metadata["Name"].lower().replace("_", "-")
            for distribution in importlib.metadata.distributions(path=[str(staging)])
        }
        if not set(RUNTIME_PACKAGES).issubset(installed):
            raise EncoderError("ENCODER_DEPENDENCIES", "The automatically installed CPU encoder dependencies are incomplete")
        (staging / ".ready").touch()
        try:
            staging.rename(RUNTIME)
        except OSError:
            if not (RUNTIME / ".ready").is_file():
                raise
        if not all(dependency_versions().values()):
            raise EncoderError("ENCODER_DEPENDENCIES", "CPU encoder dependencies are incomplete")
        return True
    except subprocess.TimeoutExpired as exc:
        raise EncoderError("ENCODER_DEPENDENCIES", "Automatic installation of CPU encoder dependencies timed out") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prepare_weights(mode: str, timeout: float = DOWNLOAD_TIMEOUT) -> dict[str, Any]:
    root = WEIGHTS_ROOT
    target = _snapshot(root)
    started = time.monotonic()
    if mode not in MODES or mode == "disabled":
        raise EncoderError("ENCODER_DISABLED", "The local encoder is disabled for this run")
    runtime_installed = _prepare_runtime() if mode == "download" else False
    if not all(dependency_versions().values()):
        raise EncoderError("ENCODER_DEPENDENCIES", "CPU encoder dependencies are missing from the cache")
    if _complete(target):
        return {"weight_cache_hit": True, "runtime_installed": runtime_installed, "asset_bytes": sum(
            (target / name).stat().st_size for name in FILES
        ), "prepare_seconds": round(time.monotonic() - started, 3)}
    if mode != "download":
        raise EncoderError("ENCODER_CACHE_MISSING", "Encoder weights are missing from the cache. Use --pretrained download to prepare them.")
    if timeout < 1:
        raise EncoderError("ENCODER_DOWNLOAD_TIMEOUT", "Insufficient time remains to download encoder weights")
    environment = _public_environment()
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["HF_HUB_DISABLE_XET"] = "1"
    if (RUNTIME / ".ready").is_file():
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (environment.get("PYTHONPATH"), str(RUNTIME)))
        )
    print(f"[C] Downloading the pinned {REPO} weights to {root}.", flush=True)
    deadline = time.monotonic() + timeout
    with subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--download", str(root)],
        env=environment, stderr=subprocess.DEVNULL,
    ) as download:
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EncoderError("ENCODER_DOWNLOAD_TIMEOUT", "Encoder weight download timed out")
                try:
                    exit_code = download.wait(timeout=min(30.0, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if remaining > 30.0:
                        blob = root / "hub" / ("models--" + REPO.replace("/", "--")) / "blobs"
                        for candidate in (blob / WEIGHT_SHA256, blob / f"{WEIGHT_SHA256}.incomplete"):
                            try:
                                downloaded = candidate.stat().st_size
                                break
                            except FileNotFoundError:
                                downloaded = 0
                        print(
                            f"[C] Encoder download progress: {downloaded / 1_000_000:.0f} / "
                            f"{WEIGHT_BYTES / 1_000_000:.0f} MB.",
                            flush=True,
                        )
        finally:
            if download.poll() is None:
                download.kill()
    if exit_code != 0 or not _complete(target):
        raise EncoderError("ENCODER_DOWNLOAD_FAILED", "Could not download the public encoder weights")
    return {"weight_cache_hit": False, "runtime_installed": runtime_installed, "asset_bytes": sum(
        (target / name).stat().st_size for name in FILES
    ), "prepare_seconds": round(time.monotonic() - started, 3)}


class LocalTextEncoder:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self.device = "cpu"
        self.versions = dependency_versions()
        self.truncated_fields = 0
        if self.versions["torch"]:
            import torch

            if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 4 * 1024**3:
                self.device = "cuda"

    def _load(self) -> None:
        if self._model is not None:
            return
        if not all(self.versions.values()):
            raise EncoderError("ENCODER_DEPENDENCIES", "Install the encoder dependencies in requirements-encoder.txt")
        target = _snapshot(WEIGHTS_ROOT)
        _verify(target)
        from transformers import AutoTokenizer, Qwen3Model
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            target, local_files_only=True, trust_remote_code=False, padding_side="right"
        )
        self._model = Qwen3Model.from_pretrained(
            target, local_files_only=True, use_safetensors=True, torch_dtype=torch.float32
        ).to(self.device).eval()

    def _encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        import torch
        from torch.nn.functional import normalize

        try:
            batch_size = int(os.environ.get("SUPERAGENT_ENCODER_BATCH_SIZE", "16"))
        except ValueError as exc:
            raise EncoderError("ENCODER_BATCH_SIZE", "Encoder batch size must be an integer from 1 to 16") from exc
        if not 1 <= batch_size <= 16:
            raise EncoderError("ENCODER_BATCH_SIZE", "Encoder batch size must be an integer from 1 to 16")

        output = []
        next_report = time.monotonic() + 20
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                ids = self._tokenizer(
                    batch, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
                self.truncated_fields += sum(len(row) == MAX_LENGTH for row in ids)
                batch_ids = [row[:MAX_LENGTH - 1] + [self._tokenizer.eos_token_id] for row in ids]
                tokens = self._tokenizer.pad(
                    {"input_ids": batch_ids}, padding=True, return_tensors="pt",
                ).to(self.device)
                positions = tokens["attention_mask"].sum(dim=1) - 1
                hidden = self._model(**tokens, use_cache=False).last_hidden_state
                pooled = hidden[torch.arange(len(batch), device=self.device), positions, :DIMENSION]
                output.append(normalize(pooled, p=2, dim=1).cpu().numpy())
                now = time.monotonic()
                if now >= next_report:
                    print(f"[Encoder] Encoded {start + len(batch)}/{len(texts)} texts", flush=True)
                    next_report = now + 20
        return np.concatenate(output, axis=0)

    def transform(
        self, matrix: np.ndarray, columns: list[str], field_mode: str,
        *, cache_dir: Path | None = None, source_hash: str | None = None,
        partition: str = "prediction",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        from data_b.modality_readers import combine_text_columns

        started = time.monotonic()
        expected = (len(matrix), DIMENSION * (len(columns) if field_mode == "separate" else 1))
        if np.prod(expected) * 4 > MAX_FEATURE_BYTES:
            raise EncoderError("ENCODER_FEATURE_LIMIT", "Encoder vectors exceed the memory limit")
        path = None
        if cache_dir is not None and source_hash is not None and partition in {"train", "validation"}:
            key = {"encoder": encoder_manifest(), "source": source_hash,
                   "columns": columns, "field_mode": field_mode, "versions": self.versions,
                   "device": self.device, "dtype": "float32", "partition": partition}
            digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
            path = cache_dir / (digest + ".npy")
        if path is not None and path.is_file():
            try:
                if path.stat().st_size > np.prod(expected) * 4 + 1024:
                    raise ValueError("invalid cache size")
                result = np.load(path, allow_pickle=False)
                if result.shape != expected or result.dtype != np.float32 or not np.isfinite(result).all():
                    raise ValueError("invalid array")
            except (OSError, ValueError, EOFError):
                pass
            else:
                return result, {"partition": partition, "cache_hit": True, "rows": len(result),
                                "seconds": round(time.monotonic() - started, 3)}
        self.truncated_fields = 0
        if field_mode == "separate":
            result = np.concatenate(
                [self._encode(matrix[:, index].tolist()) for index in range(len(columns))], axis=1
            )
        else:
            result = self._encode(combine_text_columns(matrix, columns).tolist())
        if result.shape != expected or result.dtype != np.float32 or not np.isfinite(result).all():
            raise EncoderError("ENCODER_OUTPUT_INVALID", "The encoder returned invalid vectors")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                np.save(stream, result, allow_pickle=False)
                temp = Path(stream.name)
            os.replace(temp, path)
        return result, {"partition": partition, "cache_hit": False, "rows": len(result),
                        "truncated_fields": self.truncated_fields,
                        "seconds": round(time.monotonic() - started, 3)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", type=Path, required=True)
    try:
        _download(parser.parse_args().download)
    except Exception as exc:
        print(f"[C] The encoder download subprocess failed ({type(exc).__name__}).", flush=True)
        raise SystemExit(1) from None
