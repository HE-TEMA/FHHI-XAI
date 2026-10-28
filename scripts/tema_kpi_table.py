"""Build application-schema KPI windows from fresh Kubetail rows.

The application calls full-minus-backward 'global_gmm_time_s'. This is a
compatibility residual, not an independently measured GMM fit duration.
"""
import csv
import json
import re
from collections import defaultdict
from statistics import mean
from scripts.tema_log_reports import split_operations, timing_record

FIELDS = ['model', 'entity_type', 'scope', 'aggregation', 'window_size', 'window_start', 'window_end',
          'prediction_time_s', 'global_lcrp_time_s', 'global_gmm_time_s', 'global_total_time_s',
          'attribution_single_time_s', 'allowed_time_for_global_explanations_s', 'layer',
          'attribution_single_prediction_ratio', 'allowed_global_total_ratio',
          'number_of_data_points', 'dataset_local_explanations_time_s', 'global_vs_dataset_local_ratio',
          'start_timestamp', 'end_timestamp', 'prediction_source', 'gmm_source', 'attribution_source']
TIMES = ['prediction_time_s', 'global_lcrp_time_s', 'global_gmm_time_s', 'global_total_time_s',
         'attribution_single_time_s', 'allowed_time_for_global_explanations_s']
FORWARD = re.compile(r'Forward pass time:\s*([\d.eE+-]+)\s*ms')
LAYER = re.compile(r'(?:Flood PCX layer|(?:PCX )?[Ll]ayer)\s*[:=]\s*([\w.]+)')


def ratio(a, b):
    return a / b if a is not None and b is not None and b != 0 else None


def mean_complete(rows, key):
    values = [r.get(key) for r in rows]
    return mean(values) if values and all(v is not None for v in values) else None


def derived(row, n):
    row['number_of_data_points'] = n
    attr = row['attribution_single_time_s']
    row['dataset_local_explanations_time_s'] = n * attr if n is not None and attr is not None else None
    row['attribution_single_prediction_ratio'] = ratio(attr, row['prediction_time_s'])
    row['allowed_global_total_ratio'] = ratio(row['allowed_time_for_global_explanations_s'], row['global_total_time_s'])
    # Screenshot direction is global / all local; application direction above is allowed / global.
    row['global_vs_dataset_local_ratio'] = ratio(row['global_total_time_s'], row['dataset_local_explanations_time_s'])
    return row


def build_tables(log_rows, build_boxes, in_window, target, window_size, dataset_sizes, allowed, layer_overrides):
    if window_size <= 0:
        raise ValueError('KPI_WINDOW_SIZE must be positive')
    for kind, size in dataset_sizes.items():
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
            raise ValueError('Dataset sizes must be positive integers or None: ' + kind)
    operations, _, _ = split_operations(log_rows)
    raw, windows, summaries = [], [], []
    for kind, model, entity in [('flood', 'pidnet', 'FloodSegmentation'), ('object_detection', 'yolo', 'PersonVehicleDetection')]:
        eligible = [op for op in operations[kind] if in_window(op['end_timestamp']) and
                    timing_record(op['rows'], op['start_timestamp'], op['end_timestamp'], 'image')]
        selected = eligible[-target:]
        groups = defaultdict(list)
        for index, op in enumerate(selected, 1):
            record = timing_record(op['rows'], op['start_timestamp'], op['end_timestamp'], 'image')
            forwards = [float(m.group(1)) / 1000 for r in op['rows'] for m in [FORWARD.search(r['message'])] if m]
            layers = sorted({m.group(1) for r in op['rows'] for m in [LAYER.search(r['message'])] if m})
            boxes = build_boxes(op['rows']) if kind == 'object_detection' else []
            complete_box_count = sum(bool(re.search(r'AFTER (?:TILE \d+ )?BOX \d+\b', r['message'])) for r in op['rows'])
            # Avoid treating a partial set of box timings as the single-box average.
            single = mean(b['box_full_pcx_explanation_time_s'] for b in boxes) if boxes and len(boxes) == complete_box_count else None
            if kind == 'flood': single = record['full_attribution_time_s']
            row = dict(model=model, entity_type=entity, scope='image', aggregation='raw', window_size=1,
                       window_start=index, window_end=index, prediction_time_s=forwards[0] if len(forwards) == 1 else None,
                       global_lcrp_time_s=record['local_explanation_time_s'],
                       global_gmm_time_s=max(record['full_attribution_time_s'] - record['local_explanation_time_s'], 0),
                       global_total_time_s=record['full_attribution_time_s'], attribution_single_time_s=single,
                       allowed_time_for_global_explanations_s=allowed[model],
                       layer='; '.join(layers) if layers else layer_overrides.get(model),
                       start_timestamp=op['start_timestamp'], end_timestamp=op['end_timestamp'],
                       prediction_source='standalone Forward pass time (ms converted to s)' if len(forwards) == 1 else 'unavailable: no unique standalone Forward pass time',
                       gmm_source='application-compatible residual: full attribution minus backward; not measured GMM',
                       attribution_source='per-image full attribution' if kind == 'flood' else 'mean full attribution per complete box')
            derived(row, dataset_sizes.get(model))
            raw.append(row)
            groups[row['layer']].append(row)
        if not groups:
            empty = {field: None for field in FIELDS}
            empty.update(model=model, entity_type=entity, scope='image', aggregation='selected_sample_mean',
                         window_size=0, selected_images=0,
                         allowed_time_for_global_explanations_s=allowed[model], layer=layer_overrides.get(model),
                         prediction_source='no timed images in selected period',
                         gmm_source='unavailable: no timed images in selected period',
                         attribution_source='unavailable: no timed images in selected period')
            derived(empty, dataset_sizes.get(model))
            summaries.append(empty)
        for layer, rows in groups.items():
            def aggregate(block, start, end, aggregation):
                row = {key: block[0].get(key) for key in FIELDS}
                row.update({field: mean_complete(block, field) for field in TIMES})
                row.update(aggregation=aggregation, window_size=len(block), window_start=start, window_end=end,
                           start_timestamp=block[0]['start_timestamp'], end_timestamp=block[-1]['end_timestamp'])
                # Same behavior as build_kpi_record: ratios of averaged durations.
                derived(row, dataset_sizes.get(model))
                return row
            for start in range(0, len(rows) - window_size + 1, window_size):
                block = rows[start:start + window_size]
                windows.append(aggregate(block, start+1, start+window_size, 'avg_'+str(window_size)))
            summary = aggregate(rows, 1, len(rows), 'selected_sample_mean')
            summary['selected_images'] = len(rows)
            summaries.append(summary)
    return raw, windows, summaries


def save_tables(output, raw, windows, summaries):
    for name, rows in [('kpi_schema_per_image', raw), ('kpi_schema_windows', windows), ('kpi_screenshot_table', summaries)]:
        fields = FIELDS + (['selected_images'] if name == 'kpi_screenshot_table' else [])
        with (output / (name+'.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        (output / (name+'.json')).write_text(json.dumps(rows, indent=2)+'\n')
