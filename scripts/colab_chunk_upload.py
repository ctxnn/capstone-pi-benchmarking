#!/usr/bin/env python3
"""Resumable bounded-parallel uploader for Colab CLI's base64 file endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict


def write_state(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_matches(
    session: str,
    remote_path: str,
    expected_bytes: int,
    expected_sha256: str,
    verification_lock: threading.Lock,
) -> bool:
    script = f"""
import hashlib
from pathlib import Path

path = Path({remote_path!r})
if not path.is_file() or path.stat().st_size != {expected_bytes}:
    print("CHUNK_MATCH=0")
else:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    print("CHUNK_MATCH=" + str(int(digest.hexdigest() == {expected_sha256!r})))
"""
    try:
        with verification_lock:
            completed = subprocess.run(
                ["colab", "exec", "-s", session, "--timeout", "90"],
                input=script,
                capture_output=True,
                text=True,
                timeout=120,
            )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0 and "CHUNK_MATCH=1" in completed.stdout


def upload(
    session: str,
    path: Path,
    remote_dir: str,
    retries: int,
    timeout_seconds: int,
    verification_lock: threading.Lock,
) -> Dict[str, object]:
    remote_path = f"{remote_dir.rstrip('/')}/{path.name}"
    expected_bytes = path.stat().st_size
    expected_sha256 = sha256(path)
    if remote_matches(
        session,
        remote_path,
        expected_bytes,
        expected_sha256,
        verification_lock,
    ):
        return {
            "status": "complete",
            "attempts": 0,
            "remote_path": remote_path,
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            "note": "preexisting remote chunk verified",
        }
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            completed = subprocess.run(
                ["colab", "upload", "-s", session, str(path), remote_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            if remote_matches(
                session,
                remote_path,
                expected_bytes,
                expected_sha256,
                verification_lock,
            ):
                return {
                    "status": "complete",
                    "attempts": attempt,
                    "remote_path": remote_path,
                    "bytes": expected_bytes,
                    "sha256": expected_sha256,
                    "note": "remote chunk verified after client timeout",
                }
            last_error = (
                f"colab upload exceeded the {timeout_seconds}-second per-chunk timeout"
            )
            time.sleep(min(5 * attempt, 20))
            continue
        if completed.returncode == 0:
            return {
                "status": "complete",
                "attempts": attempt,
                "remote_path": remote_path,
                "bytes": expected_bytes,
                "sha256": expected_sha256,
            }
        if remote_matches(
            session,
            remote_path,
            expected_bytes,
            expected_sha256,
            verification_lock,
        ):
            return {
                "status": "complete",
                "attempts": attempt,
                "remote_path": remote_path,
                "bytes": expected_bytes,
                "sha256": expected_sha256,
                "note": "remote chunk verified after nonzero client exit",
            }
        last_error = (completed.stderr + "\n" + completed.stdout).strip()
        time.sleep(min(5 * attempt, 20))
    return {
        "status": "failed",
        "attempts": retries,
        "remote_path": remote_path,
        "error": last_error[-2000:],
    }


def keep_session_alive(session: str, stop: threading.Event, interval: int) -> None:
    while not stop.is_set():
        try:
            subprocess.run(
                ["colab", "exec", "-s", session, "--timeout", "60"],
                input="import time; print('chunk-upload-keepalive', time.time())\n",
                capture_output=True,
                text=True,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            pass
        stop.wait(interval)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--remote-dir", default="/content")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--skip-through", type=int, default=-1)
    parser.add_argument("--keepalive-seconds", type=int, default=180)
    args = parser.parse_args()

    paths = sorted(path for path in args.chunks.iterdir() if path.is_file())
    state: Dict[str, object]
    if args.state.exists():
        state = json.loads(args.state.read_text())
    else:
        state = {"schema_version": 1, "session": args.session, "chunks": {}}
    chunks: Dict[str, object] = state["chunks"]  # type: ignore[assignment]
    for index, path in enumerate(paths):
        if index <= args.skip_through:
            chunks[path.name] = {
                "status": "complete",
                "attempts": 1,
                "remote_path": f"{args.remote_dir.rstrip('/')}/{path.name}",
                "note": "uploaded before resumable transfer",
            }
    pending = [path for path in paths if chunks.get(path.name, {}).get("status") != "complete"]  # type: ignore[union-attr]
    lock = threading.Lock()
    verification_lock = threading.Lock()
    print(f"chunks={len(paths)} pending={len(pending)} workers={args.workers}", flush=True)
    stop_keepalive = threading.Event()
    keepalive = threading.Thread(
        target=keep_session_alive,
        args=(args.session, stop_keepalive, args.keepalive_seconds),
        daemon=True,
    )
    keepalive.start()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    upload,
                    args.session,
                    path,
                    args.remote_dir,
                    args.retries,
                    args.timeout_seconds,
                    verification_lock,
                ): path
                for path in pending
            }
            for completed_count, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                result = future.result()
                with lock:
                    chunks[path.name] = result
                    write_state(args.state, state)
                print(
                    f"[{completed_count}/{len(pending)}] {path.name}: {result['status']}",
                    flush=True,
                )
    finally:
        stop_keepalive.set()
        keepalive.join(timeout=5)
    failed = [name for name, result in chunks.items() if result.get("status") != "complete"]  # type: ignore[union-attr]
    print(f"complete={len(paths) - len(failed)} failed={len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
