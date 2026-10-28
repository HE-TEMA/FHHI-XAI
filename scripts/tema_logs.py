#!/usr/bin/env python3
"""Capture the TEMA browser log console and summarize logged timing values.

Setup: python -m pip install playwright && python -m playwright install chromium
Capture: python scripts/tema_logs.py --headed --login-wait 45 --duration 300
Offline: python scripts/tema_logs.py --input saved.log
KPI files: python scripts/tema_logs.py --kpi-dir output/kpi_logs
Fetch KPI files: python scripts/tema_logs.py --minio

Only rendered text is captured. Select the log panel with --selector if the
default body includes navigation. Virtualized/rapidly rotating lines can be
missed; this is not an archival Kubernetes log exporter. Naive timestamps retain
the server's unspecified timezone; they are never replaced with capture time.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import time


DEFAULT_URL = (
    "https://tema-project.ddns.net/logs/console?source=tema%2Fdeployments%2Ftfa02"
    "&container=tema%2Ftfa02-ff95f9555-82wz6%2Ftfa02-container"
    "&container=tema%2Ftfa02-58b8b96d96-568lf%2Ftfa02-container"
    "&os=linux&arch=amd64&node=auth-0z8uf"
    "&container=tema%2Ftfa02-567db959bc-9qq6q%2Ftfa02-container"
)
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:\d{2})?")
TASK = re.compile(r"Task ([\w-]+): (.*)")
TIMING = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z _/-]*?(?:time|duration|elapsed))\s*[:=]\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>milliseconds|seconds|minutes|ms|s|min)\b",
    re.I,
)
KPI = re.compile(r"\b([a-z][a-z0-9_]*_time_s)\s*[:=]\s*(\d+(?:\.\d+)?)\b")


def added_lines(previous, current):
    """Find the longest suffix/prefix overlap, preserving repeated log lines."""
    for size in range(min(len(previous), len(current)), 0, -1):
        if previous[-size:] == current[:size]:
            return current[size:]
    return current


def capture(args, destination):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Install playwright and Chromium using the setup command in --help.") from exc
    previous = []
    with destination.open("w", encoding="utf-8") as raw, \
            destination.with_suffix(".capture.jsonl").open("w", encoding="utf-8") as audit, \
            sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        context = browser.new_context(storage_state=args.storage_state or None)
        page = context.new_page()
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
            if args.login_wait:
                print(f"Waiting {args.login_wait:g}s for browser login…", flush=True)
                page.wait_for_timeout(args.login_wait * 1000)
            start = time.monotonic()
            while time.monotonic() - start < args.duration:
                lines = page.locator(args.selector).inner_text(timeout=15000).splitlines()
                fresh = added_lines(previous, lines)
                for line in fresh:
                    raw.write(line + "\n")
                    audit.write(json.dumps({"captured_at": datetime.now(timezone.utc).isoformat(),
                                            "text": line}) + "\n")
                raw.flush()
                audit.flush()
                previous = lines
                page.wait_for_timeout(args.interval * 1000)
        finally:
            browser.close()


def summarize(values):
    ordered = sorted(values)
    # Linear interpolation, including a single observation.
    rank = (len(ordered) - 1) * .95
    lo = int(rank)
    p95 = ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (rank - lo)
    return dict(count=len(values), total_s=sum(values), mean_s=statistics.mean(values),
                median_s=statistics.median(values), min_s=min(values), max_s=max(values), p95_s=p95)


def analyze(source, output):
    metrics, stages, intervals = {}, {}, []
    timestamped = 0
    for number, line in enumerate(source.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        match = STAMP.search(line)
        stamp = None
        if match:
            try:
                stamp = datetime.fromisoformat(match[0].replace(",", ".").replace("Z", "+00:00"))
                timestamped += 1
            except ValueError:
                pass
        for match in TIMING.finditer(line):
            unit = match["unit"].lower()
            scale = .001 if unit in ("ms", "milliseconds") else 60 if unit in ("min", "minutes") else 1
            metrics.setdefault(match["label"].strip().lower(), []).append(float(match["value"]) * scale)
        for match in KPI.finditer(line):
            metrics.setdefault(match[1], []).append(float(match[2]))
        task = TASK.search(line)
        if task and stamp:
            task_id, stage = task.groups()
            if task_id in stages:
                old_stamp, old_stage = stages[task_id]
                try:
                    elapsed = (stamp - old_stamp).total_seconds()
                except TypeError:
                    elapsed = -1  # Mixed aware/naive timestamps are not comparable.
                if elapsed >= 0:
                    intervals.append(dict(task_id=task_id, from_stage=old_stage, to_stage=stage,
                                          start=old_stamp.isoformat(), end=stamp.isoformat(), elapsed_s=elapsed))
            stages[task_id] = stamp, stage
    output.mkdir(parents=True, exist_ok=True)
    with (output / "task_intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "from_stage", "to_stage", "start", "end", "elapsed_s"])
        writer.writeheader()
        writer.writerows(intervals)
    report = dict(source=str(source), timestamped_lines=timestamped, tasks_with_timestamps=len(stages),
                  task_intervals=len(intervals), timing_statistics={k: summarize(v) for k, v in metrics.items()},
                  notes=["Task intervals measure consecutive logged stages, not guaranteed end-to-end runtime.",
                         "Timestamp timezone is retained as logged; naive timestamps have unspecified timezone.",
                         "Repeated browser snapshots without overlap may duplicate lines; virtualized lines may be missed."])
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not timestamped and not metrics:
        print("No timestamped logs or timing values found. Check login, selected containers and --selector.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help="Console URL; override if pod/container IDs change")
    parser.add_argument("--input", type=Path, help="Analyze an existing text log without opening a browser")
    parser.add_argument("--kpi-dir", type=Path, help="Calculate exact KPI metrics from local CSV/.txt files")
    parser.add_argument("--minio", action="store_true", help="Download and calculate KPI files from configured MinIO (requires minio package)")
    parser.add_argument("--output", type=Path, default=Path("output/tema_logs"))
    parser.add_argument("--duration", type=float, default=60, help="Capture duration in seconds")
    parser.add_argument("--interval", type=float, default=1, help="Browser polling interval in seconds")
    parser.add_argument("--headed", action="store_true", help="Show browser (requires a display)")
    parser.add_argument("--login-wait", type=float, default=0)
    parser.add_argument("--storage-state", help="Existing Playwright authentication state JSON")
    parser.add_argument("--selector", default="body", help="CSS selector for a single log panel")
    args = parser.parse_args()
    if sum((bool(args.input), bool(args.kpi_dir), args.minio)) > 1:
        parser.error("Choose only one of --input, --kpi-dir, --minio")
    if args.duration <= 0 or args.interval <= 0 or args.login_wait < 0:
        parser.error("duration/interval must be positive and login-wait nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.input or args.output / (datetime.now(timezone.utc).strftime("console_%Y%m%dT%H%M%S_%fZ") + ".log")
    try:
        if args.kpi_dir or args.minio:
            from tema_kpi import calculate, download
            directory = args.kpi_dir if args.kpi_dir else download(args.output / "downloads")
            calculate(directory, args.output)
            return
        if not args.input:
            try:
                capture(args, source)
            except KeyboardInterrupt:
                print("Capture stopped; analyzing saved lines.")
        analyze(source, args.output)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
