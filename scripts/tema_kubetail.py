#!/usr/bin/env python3
"""Fetch fresh Kubetail GraphQL logs and report flood/object timings. No cached KPI inputs."""
import argparse
import json
import sys
from pathlib import Path

from tema_log_reports import analyze_logs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', help='Inclusive box/image end timestamp, ISO 8601')
    parser.add_argument('--end', help='Inclusive box/image end timestamp, ISO 8601')
    parser.add_argument('--pod', help='Pod name; otherwise discover current TFA02 pod')
    parser.add_argument('--max-pages', type=int, default=60)
    parser.add_argument('--page-size', type=int, default=1000)
    parser.add_argument('--recent', type=int, default=100, help='Number of complete timed IMAGES per type within the window (default: 100 each)')
    args = parser.parse_args()
    if min(args.max_pages, args.page_size, args.recent) <= 0:
        parser.error('Page limits and recent count must be positive')
    notebook = json.loads((ROOT / 'tfa02_log_kpis.ipynb').read_text())
    namespace = {}
    # Reuse the notebook's connection configuration, API queries and parser.
    # These cells contain settings and definitions, not cached log/results data.
    for index in (1, 2, 3):
        exec(compile(''.join(notebook['cells'][index]['source']), str(ROOT / 'tfa02_log_kpis.ipynb') + ':cell' + str(index), 'exec'), namespace)
    namespace.update(START_TIME=namespace['parse_timestamp'](args.start) if args.start else None,
                     END_TIME=namespace['parse_timestamp'](args.end) if args.end else None,
                     POD_NAME=args.pod, MAX_PAGES=args.max_pages, PAGE_SIZE=args.page_size)
    if namespace['START_TIME'] and namespace['END_TIME'] and namespace['START_TIME'] > namespace['END_TIME']:
        parser.error('--start must not be later than --end')
    output = ROOT / 'output' / 'kubetail' / namespace['RUN_TIMESTAMP']
    print('Fetching fresh Kubetail logs from ' + namespace['GRAPHQL_URL'], flush=True)
    # A network failure stops the run; no saved CSV, MinIO or browser fallback.
    pod, rows = namespace['collect_log_rows_for_recent_boxes'](args.recent)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'raw_logs.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    (output / 'provenance.json').write_text(json.dumps(dict(
        source='live Kubetail GraphQL', endpoint=namespace['GRAPHQL_URL'], pod=pod,
        fetched_rows=len(rows), start=args.start, end=args.end,
        page_size=args.page_size, max_pages=args.max_pages), indent=2) + '\n')
    reports = analyze_logs(rows, output, namespace['build_box_records'], namespace['in_time_window'], args.recent)
    if any(r['status'] != 'complete' for r in reports.values()):
        print('Requested image counts were not reached. Reports contain only available images; widen the time window, increase --max-pages, or select an older --pod.')
    print('Reports and raw Kubetail logs: ' + str(output))


if __name__ == '__main__':
    main()
