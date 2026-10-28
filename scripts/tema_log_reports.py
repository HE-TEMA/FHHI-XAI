"""Separate log timing calls by explicit operation markers; never infer boxes from calls."""
import csv
import json
import re
from pathlib import Path
from statistics import mean

NUMBER = r'(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
# Three named log measurements on the same line; case and field order do not matter.
# Attribution Time and Full Attribution Time denote the same logged field here.
TIMING = re.compile(
    r'^(?=.*\bPrediction\s+Time\s*[:=]\s*' + NUMBER + r')'
    r'(?=.*\bBackward\s+Time\s*[:=]\s*' + NUMBER + r')'
    r'(?=.*\b(?:Full\s+)?Attribution\s+Time\s*[:=]\s*' + NUMBER + r')',
    re.I,
)
# Editable patterns for deployed versions. PIDNet alone is not enough to identify flood (also used for fire).
OP_START = re.compile(r'BEFORE EXPLAIN\s+(\w+)|\b(FLOOD_SEG|PERSON_VEHICLE|FIRE_SEG)\s+START\b', re.I)
OP_END = re.compile(r'AFTER EXPLAIN\s+(\w+)|\b(FLOOD_SEG|PERSON_VEHICLE|FIRE_SEG)\s+(?:END|NO (?:VALID )?DETECTIONS)\b', re.I)


def category(name):
    name = name.lower()
    if 'flood' in name: return 'flood'
    if 'person' in name or 'vehicle' in name or 'detection' in name: return 'object_detection'
    if 'fire' in name: return 'fire'
    return 'unknown'


def split_operations(rows):
    current = None
    completed = {'flood': [], 'object_detection': [], 'fire': [], 'unknown': []}
    unassigned = []
    incomplete = []
    for row in rows:
        message = row['message']
        start, end = OP_START.search(message), OP_END.search(message)
        if start:
            kind = category(start.group(1) or start.group(2))
            # BEFORE EXPLAIN and model START may describe the same operation.
            if current and (current['category'] != kind or start.group(1)):
                incomplete.append(current)
                current = None
            if current is None:
                current = dict(category=kind, rows=[], start_timestamp=row['timestamp'])
        if current:
            current['rows'].append(row)
        elif TIMING.search(message):
            unassigned.append(row)
        if end and current and category(end.group(1) or end.group(2)) == current['category']:
            current['end_timestamp'] = row['timestamp']
            completed[current['category']].append(current)
            current = None
    if current: incomplete.append(current)
    return completed, unassigned, incomplete


def timing_record(rows, start, end, scope):
    timings = [tuple(map(float, m.groups())) for row in rows for m in [TIMING.search(row['message'])] if m]
    if not timings: return None
    return dict(start_timestamp=start, end_timestamp=end, scope=scope, num_timing_lines=len(timings),
                prediction_time_s=sum(t[0] for t in timings), local_explanation_time_s=sum(t[1] for t in timings),
                full_attribution_time_s=sum(t[2] for t in timings))


def write_report(records, output, name, source):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fields = ['start_timestamp','end_timestamp','scope','num_timing_lines','prediction_time_s','local_explanation_time_s','full_attribution_time_s']
    with (output / (name+'_records.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    local=sum(r['local_explanation_time_s'] for r in records)
    ratios=[r['local_explanation_time_s']/r['prediction_time_s'] for r in records if r['prediction_time_s']>0]
    summary=dict(source=source, records=len(records), scopes=sorted(set(r['scope'] for r in records)),
        avg_prediction_time_s=mean(r['prediction_time_s'] for r in records) if records else None,
        avg_local_explanation_time_s=mean(r['local_explanation_time_s'] for r in records) if records else None,
        avg_full_attribution_time_s=mean(r['full_attribution_time_s'] for r in records) if records else None,
        local_vs_prediction_ratio=mean(ratios) if ratios else None,
        full_attribution_vs_all_local_ratio=sum(r['full_attribution_time_s'] for r in records)/local if local else None)
    (output / (name+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(name, json.dumps(summary,indent=2))
    return summary


def analyze_logs(rows, output, build_boxes, in_window, recent=100):
    operations, unassigned, incomplete = split_operations(rows)
    reports = {}
    for kind in ('flood','object_detection'):
        images=[]; boxes=[]; calls=[]
        eligible = [op for op in operations[kind]
                    if in_window(op['end_timestamp']) and any(TIMING.search(r['message']) for r in op['rows'])]
        selected = eligible[-recent:]
        for op in selected:
            record=timing_record(op['rows'],op['start_timestamp'],op['end_timestamp'],'image')
            if record: images.append(record)
            if kind == 'object_detection':
                for box in build_boxes(op['rows']):
                    boxes.append(dict(start_timestamp=box['start_timestamp'],end_timestamp=box['end_timestamp'],scope='box',
                        num_timing_lines=box['num_timing_lines'],prediction_time_s=box['box_prediction_time_s'],
                        local_explanation_time_s=box['box_local_explanation_time_s'],full_attribution_time_s=box['box_full_pcx_explanation_time_s']))
            for row in op['rows']:
                record=timing_record([row],row['timestamp'],row['timestamp'],'attribution_call')
                if record: calls.append(record)
        # Box parser returns newest-first per image; restore global row order.
        end_order = {row['timestamp']: index for index, row in enumerate(rows)}
        boxes.sort(key=lambda box: end_order[box['end_timestamp']])
        # Image selection was performed above; retain all boxes in those images.
        reports[kind]=write_report(images, output, kind+'_images', 'backend log timing sums; image boundaries explicit')
        reports[kind].update(requested_images=recent, available_images_in_window=len(eligible),
                             status='complete' if len(images)==recent else 'insufficient_images_in_window',
                             first_selected_start=images[0]['start_timestamp'] if images else None,
                             last_selected_end=images[-1]['end_timestamp'] if images else None)
        (Path(output)/(kind+'_images_summary.json')).write_text(json.dumps(reports[kind],indent=2)+'\n')
        print(f"{kind}: selected {len(images)}/{recent} images in requested window ({reports[kind]['status']})")
        if kind == 'object_detection':
            write_report(boxes,output,kind+'_boxes','backend log timing sums; box boundaries explicit')
        write_report(calls,output,kind+'_calls','individual attribution calls inside completed image operations; not boxes')
    diagnostics=dict(unassigned_timing_lines=len(unassigned), incomplete_operations=len(incomplete),
        completed_operations={k:len(v) for k,v in operations.items()},
        note='Single-stream operation grouping assumes jobs do not interleave. Concurrent workers require worker/task IDs. Log prediction sums measure attribution forwards, not the standalone prediction_time_s in stored KPI records.')
    (Path(output)/'classification_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print('Classification diagnostics:',diagnostics)
    return reports
