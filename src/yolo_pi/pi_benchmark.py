from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .benchmark import summarize
from .detector import UltralyticsDetector
from .input_sources import create_input_source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    root = path if path.is_dir() else path.parent
    files = [path] if path.is_file() else sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
        and item.name != ".DS_Store"
    )
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_pi_model() -> Optional[str]:
    path = Path("/proc/device-tree/model")
    if not path.exists():
        return None
    return path.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")


def _temperature_c() -> Optional[float]:
    thermal_root = Path("/sys/class/thermal")
    for path in sorted(thermal_root.glob("thermal_zone*/temp")):
        try:
            value = float(path.read_text(encoding="utf-8").strip())
            return value / 1000.0 if value > 1000 else value
        except (OSError, ValueError):
            continue
    return None


def _throttled_status() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["vcgencmd", "get_throttled"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value or None


def environment_manifest() -> Dict[str, Any]:
    pi_model = _read_pi_model()
    try:
        import psutil

        memory_bytes = psutil.virtual_memory().total
        psutil_version = psutil.__version__
    except ImportError:
        memory_bytes = None
        psutil_version = None
    return {
        "captured_at_epoch_ns": time.time_ns(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "memory_total_bytes": memory_bytes,
        "psutil": psutil_version,
        "raspberry_pi_model": pi_model,
        "is_raspberry_pi": bool(pi_model and "Raspberry Pi" in pi_model),
        "temperature_c_at_start": _temperature_c(),
        "throttled_at_start": _throttled_status(),
    }


class SystemSampler:
    def __init__(self) -> None:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError(
                "psutil is required for Pi resource measurements; install the 'pi' extra"
            ) from exc
        self.psutil = psutil
        self.process = psutil.Process()
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)

    def sample(self) -> Dict[str, Optional[float]]:
        memory = self.process.memory_info()
        swap = self.psutil.swap_memory()
        return {
            "system_cpu_percent": float(self.psutil.cpu_percent(interval=None)),
            "process_cpu_percent": float(self.process.cpu_percent(interval=None)),
            "rss_mb": float(memory.rss) / (1024 * 1024),
            "swap_used_mb": float(swap.used) / (1024 * 1024),
            "temperature_c": _temperature_c(),
        }


def _summary_or_none(values: Iterable[Optional[float]]) -> Optional[Dict[str, float]]:
    present = [float(value) for value in values if value is not None and math.isfinite(value)]
    return summarize(present) if present else None


def _power_summary(path: Optional[Path], started_ns: int, finished_ns: int) -> Optional[Dict[str, float]]:
    if path is None:
        return None
    values: List[float] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        sample = json.loads(raw_line)
        timestamp_ns = int(sample["timestamp_ns"])
        if started_ns <= timestamp_ns <= finished_ns:
            values.append(float(sample["power_w"]))
    return summarize(values) if values else None


def load_fixed_inputs(manifest_path: Path, workspace_root: Path) -> List[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs: List[Path] = []
    for entry in manifest["inputs"]:
        path = workspace_root / str(entry["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        expected = entry.get("sha256")
        if expected and sha256(path) != expected:
            raise ValueError(f"fixed input checksum mismatch: {path}")
        inputs.append(path)
    if not inputs:
        raise ValueError("fixed input manifest contains no inputs")
    return inputs


def _resolve_artifact(entry: Mapping[str, Any], workspace_root: Path) -> Path:
    path = Path(str(entry["path"]))
    return path if path.is_absolute() else workspace_root / path


def configure_threads(threads: int) -> None:
    if threads <= 0:
        raise ValueError("threads must be positive")
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except (ImportError, RuntimeError):
        pass


def run_row(
    row: Mapping[str, Any],
    registry_entry: Mapping[str, Any],
    inputs: Sequence[Path],
    *,
    warmup_runs: int,
    repetitions: int,
    workspace_root: Path,
    power_jsonl: Optional[Path] = None,
) -> Dict[str, Any]:
    model_path = _resolve_artifact(registry_entry, workspace_root)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    expected_sha = registry_entry.get("sha256")
    artifact_sha = sha256(model_path) if model_path.is_file() else None
    if expected_sha and artifact_sha != expected_sha:
        raise ValueError(f"model checksum mismatch: {model_path}")
    artifact_tree_sha = tree_sha256(model_path)
    expected_tree_sha = registry_entry.get("tree_sha256")
    if expected_tree_sha and artifact_tree_sha != expected_tree_sha:
        raise ValueError(f"model tree checksum mismatch: {model_path}")

    threads = int(row.get("threads", 4))
    configure_threads(threads)
    detector = UltralyticsDetector(
        model_path,
        imgsz=int(row["imgsz"]),
        confidence=float(row.get("confidence", 0.35)),
        device=row.get("device"),
        runtime_threads=threads,
        runtime_precision=str(row.get("precision", "FP32")),
    )
    expected_backend = {
        "pytorch": "pytorch-",
        "onnx runtime": "onnxruntime-",
        "openvino": "openvino-",
        "mnn": "mnn-high-precision-",
        "ncnn": "ncnn-",
        "litert": "litert-",
    }.get(str(row.get("runtime", "")).casefold())
    if expected_backend and not detector.backend_name.startswith(expected_backend):
        raise RuntimeError(
            f"row requests {row['runtime']} but resolved backend is {detector.backend_name}"
        )
    source_type = str(row.get("source", "images"))
    input_source = create_input_source(
        source_type,
        inputs=inputs,
        camera_config=dict(row.get("camera", {})),
    )
    input_source.start()
    try:
        for index in range(warmup_runs):
            frame = input_source.next_frame()
            detector.detect(
                frame.payload,
                frame_id=-(index + 1),
                captured_ns=frame.captured_ns,
                source=frame.source,
            )

        sampler = SystemSampler()
        observations: List[Dict[str, Any]] = []
        throttled_before = _throttled_status()
        started_epoch_ns = time.time_ns()
        for repetition in range(repetitions):
            frame = input_source.next_frame()
            result = detector.detect(
                frame.payload,
                frame_id=repetition,
                captured_ns=frame.captured_ns,
                source=frame.source,
            )
            system = sampler.sample()
            stages = {
                name: float(result.stage_ms.get(name, 0.0))
                for name in ("preprocess", "inference", "postprocess")
            }
            camera_to_finish_ms = (
                (result.model_finished_ns - frame.captured_ns) / 1_000_000.0
                if frame.captured_ns is not None
                else None
            )
            capture_to_finish_ms = (
                (result.model_finished_ns - frame.capture_started_ns) / 1_000_000.0
                if frame.capture_started_ns is not None
                else None
            )
            frame_age_at_model_start_ms = (
                (result.model_started_ns - frame.captured_ns) / 1_000_000.0
                if frame.captured_ns is not None
                else None
            )
            observations.append(
                {
                    "repetition": repetition,
                    "source": frame.source,
                    "source_type": frame.source_type,
                    "camera_frame_id": frame.frame_id,
                    "dropped_frames_since_last": frame.dropped_since_last,
                    "detection_count": len(result.detections),
                    "capture_ms": frame.capture_ms,
                    "frame_age_at_available_ms": frame.frame_age_at_available_ms,
                    "frame_age_at_model_start_ms": frame_age_at_model_start_ms,
                    "camera_to_model_finish_ms": camera_to_finish_ms,
                    "capture_start_to_model_finish_ms": capture_to_finish_ms,
                    "wall_total_ms": result.model_call_ms,
                    "preprocess_ms": stages["preprocess"],
                    "inference_ms": stages["inference"],
                    "postprocess_ms": stages["postprocess"],
                    **system,
                }
            )
        finished_epoch_ns = time.time_ns()
    finally:
        input_source.close()

    metric_keys = (
        "wall_total_ms",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "system_cpu_percent",
        "process_cpu_percent",
        "rss_mb",
        "swap_used_mb",
        "temperature_c",
        "capture_ms",
        "frame_age_at_available_ms",
        "frame_age_at_model_start_ms",
        "camera_to_model_finish_ms",
        "capture_start_to_model_finish_ms",
    )
    summaries = {
        key: _summary_or_none(observation.get(key) for observation in observations)
        for key in metric_keys
    }
    total_mean = summaries["wall_total_ms"]["mean"]  # type: ignore[index]
    return {
        "schema_version": 1,
        "status": "complete",
        "scope": "Raspberry Pi measurement" if _read_pi_model() else "non-Pi diagnostic measurement",
        "row": dict(row),
        "artifact": {
            "registry_key": row["model_key"],
            "path": str(model_path),
            "sha256": artifact_sha,
            "tree_sha256": artifact_tree_sha,
            "resolved_backend": detector.backend_name,
        },
        "warmup_runs_excluded": warmup_runs,
        "repetitions": repetitions,
        "threads": threads,
        "source_type": source_type,
        "started_at_epoch_ns": started_epoch_ns,
        "finished_at_epoch_ns": finished_epoch_ns,
        "metrics": summaries,
        "fps_from_wall_mean": 1000.0 / total_mean if total_mean > 0 else 0.0,
        "dropped_frames_total": sum(
            int(observation["dropped_frames_since_last"])
            for observation in observations
        ),
        "power_w": _power_summary(power_jsonl, started_epoch_ns, finished_epoch_ns),
        "throttled_before": throttled_before,
        "throttled_after": _throttled_status(),
        "observations": observations,
    }


def cooldown(target_c: float, timeout_seconds: float) -> Dict[str, Any]:
    started = time.monotonic()
    samples: List[float] = []
    while True:
        temperature = _temperature_c()
        if temperature is None or temperature <= target_c:
            return {
                "target_c": target_c,
                "duration_seconds": time.monotonic() - started,
                "temperatures_c": samples,
                "reached_target": temperature is None or temperature <= target_c,
            }
        samples.append(temperature)
        if time.monotonic() - started >= timeout_seconds:
            return {
                "target_c": target_c,
                "duration_seconds": time.monotonic() - started,
                "temperatures_c": samples,
                "reached_target": False,
            }
        time.sleep(5)


def run_matrix(
    matrix_path: Path,
    registry_path: Path,
    output_dir: Path,
    *,
    row_ids: Optional[Sequence[str]] = None,
    force: bool = False,
    allow_non_pi: bool = False,
    power_jsonl: Optional[Path] = None,
    source_type: Optional[str] = None,
    camera_config: Optional[Mapping[str, Any]] = None,
    threads: Optional[int] = None,
) -> Dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    workspace_root = matrix_path.resolve().parent.parent
    environment = environment_manifest()
    if not environment["is_raspberry_pi"] and not allow_non_pi:
        raise RuntimeError("refusing to create Pi evidence on non-Raspberry-Pi hardware")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "environment.json", environment)
    inputs_manifest = workspace_root / str(matrix["fixed_inputs_manifest"])
    inputs = load_fixed_inputs(inputs_manifest, workspace_root)
    defaults = matrix.get("source_defaults", {"type": "images"})
    all_rows = []
    for original in matrix["architecture_experiments"] + matrix["runtime_experiments"]:
        row = dict(original)
        row["source"] = source_type or row.get("source") or defaults.get("type", "images")
        if row["source"] == "camera":
            merged_camera = dict(defaults.get("camera", {}))
            merged_camera.update(row.get("camera", {}))
            if camera_config:
                merged_camera.update(camera_config)
            row["camera"] = merged_camera
        row["threads"] = int(threads or row.get("threads") or matrix.get("threads", 4))
        all_rows.append(row)
    selected = [row for row in all_rows if row_ids is None or row["id"] in row_ids]
    if row_ids and {row["id"] for row in selected} != set(row_ids):
        missing = sorted(set(row_ids) - {row["id"] for row in selected})
        raise ValueError(f"unknown row IDs: {missing}")

    outcomes: Dict[str, Any] = {}
    for row in selected:
        row_path = output_dir / "rows" / f"{row['id']}.json"
        if row_path.exists() and not force:
            existing = json.loads(row_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                outcomes[row["id"]] = "skipped-complete"
                continue
        cooling = cooldown(
            float(matrix.get("cooldown_target_c", 55.0)),
            float(matrix.get("cooldown_timeout_seconds", 600.0)),
        )
        try:
            registry_entry = registry["artifacts"][row["model_key"]]
            report = run_row(
                row,
                registry_entry,
                inputs,
                warmup_runs=int(matrix["warmup_runs"]),
                repetitions=int(matrix["repetitions"]),
                workspace_root=workspace_root,
                power_jsonl=power_jsonl,
            )
            report["cooldown_before"] = cooling
            atomic_json(row_path, report)
            outcomes[row["id"]] = "complete"
        except Exception as exc:
            failure = {
                "schema_version": 1,
                "status": "failed",
                "row": row,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "cooldown_before": cooling,
            }
            atomic_json(row_path, failure)
            outcomes[row["id"]] = "failed"
    compile_tables(matrix_path, registry_path, output_dir, output_dir / "tables")
    return outcomes


def _value(report: Optional[Mapping[str, Any]], metric: str, statistic: str = "mean") -> Any:
    if not report or report.get("status") != "complete":
        return None
    summary = report["metrics"].get(metric)
    return summary.get(statistic) if summary else None


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _percent_or_none(value: Any) -> Optional[float]:
    return float(value) * 100.0 if isinstance(value, (int, float)) else None


def _write_table(output_base: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    csv_path = output_base.with_suffix(".csv")
    markdown_path = output_base.with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_tables(
    matrix_path: Path, registry_path: Path, run_dir: Path, output_dir: Path
) -> Dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    def report_for(row_id: str) -> Optional[Dict[str, Any]]:
        path = run_dir / "rows" / f"{row_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    architecture_headers = [
        "ID", "Model", "Runtime", "Resolution", "Precision", "Threads", "Source",
        "Mean inference (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)",
        "FPS", "Peak RAM (MB)", "CPU utilization mean (%)", "CPU temperature max (C)",
        "mAP50 (%)", "mAP50-95 (%)", "Recall (%)", "Status",
    ]
    architecture_rows = []
    for row in matrix["architecture_experiments"]:
        report = report_for(row["id"])
        display_row = report.get("row", row) if report else row
        quality = registry["artifacts"].get(display_row["model_key"], {}).get("quality_metrics", {})
        architecture_rows.append(
            [
                display_row["id"], display_row["model"], display_row["runtime"], display_row["imgsz"], display_row["precision"],
                display_row.get("threads", matrix.get("threads", 4)), display_row.get("source", matrix.get("source_defaults", {}).get("type", "images")),
                _value(report, "inference_ms"), _value(report, "inference_ms", "p50"),
                _value(report, "inference_ms", "p95"), _value(report, "inference_ms", "p99"),
                report.get("fps_from_wall_mean") if report and report.get("status") == "complete" else None,
                _value(report, "rss_mb", "max"), _value(report, "system_cpu_percent"),
                _value(report, "temperature_c", "max"),
                _percent_or_none(quality.get("map50")),
                _percent_or_none(quality.get("map50_95")),
                _percent_or_none(quality.get("recall")),
                report.get("status", "pending") if report else "pending",
            ]
        )
    _write_table(output_dir / "model-architecture-benchmark", architecture_headers, architecture_rows)

    runtime_headers = [
        "ID", "Model", "Runtime", "Resolution", "Precision", "Threads", "Source",
        "Capture mean (ms)", "Frame age at model start mean (ms)", "Camera to model finish mean (ms)",
        "Preprocess mean (ms)", "Inference mean (ms)", "Postprocess mean (ms)",
        "Total pipeline mean (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)", "FPS",
        "CPU utilization mean (%)", "Peak RAM (MB)", "Temperature max (C)",
        "Power mean (W)", "Status",
    ]
    runtime_rows = []
    for row in matrix["runtime_experiments"]:
        report = report_for(row["id"])
        display_row = report.get("row", row) if report else row
        power = report.get("power_w") if report and report.get("status") == "complete" else None
        runtime_rows.append(
            [
                display_row["id"], display_row["model"], display_row["runtime"], display_row["imgsz"], display_row["precision"],
                display_row.get("threads", matrix.get("threads", 4)), display_row.get("source", matrix.get("source_defaults", {}).get("type", "images")),
                _value(report, "capture_ms"), _value(report, "frame_age_at_model_start_ms"),
                _value(report, "camera_to_model_finish_ms"),
                _value(report, "preprocess_ms"), _value(report, "inference_ms"),
                _value(report, "postprocess_ms"), _value(report, "wall_total_ms"),
                _value(report, "wall_total_ms", "p50"), _value(report, "wall_total_ms", "p95"),
                _value(report, "wall_total_ms", "p99"),
                report.get("fps_from_wall_mean") if report and report.get("status") == "complete" else None,
                _value(report, "system_cpu_percent"), _value(report, "rss_mb", "max"),
                _value(report, "temperature_c", "max"),
                power.get("mean") if power else None,
                report.get("status", "pending") if report else "pending",
            ]
        )
    _write_table(output_dir / "runtime-backend-benchmark", runtime_headers, runtime_rows)
    manifest = {
        "schema_version": 1,
        "generated_at_epoch_ns": time.time_ns(),
        "matrix": str(matrix_path),
        "registry": str(registry_path),
        "run_dir": str(run_dir),
        "architecture_rows": len(architecture_rows),
        "runtime_rows": len(runtime_rows),
        "missing_values_render_as": "NA",
        "warning": "NA is never replaced with estimates. Performance values are valid Pi evidence only when row scope is Raspberry Pi measurement.",
    }
    atomic_json(output_dir / "table-manifest.json", manifest)
    return manifest
