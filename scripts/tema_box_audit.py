"""Original 100-box formulas, independent of image/schema/screenshot ratios."""
import csv
import json
from statistics import mean
from scripts.tema_log_reports import split_operations


def audit_boxes(rows, build_boxes, in_window, parse_timestamp, output, count=100):
    if count <= 0:
        raise ValueError('Box target must be positive')
    operations, _, _ = split_operations(rows)
    boxes = []
    for image_index, op in enumerate(operations['object_detection'], 1):
        # Filter each BOX end, not its containing image end.
        for box in build_boxes(op['rows']):
            if not in_window(box['end_timestamp']):
                continue
            record = dict(box)
            record['image_index_in_fetched_logs'] = image_index
            record['image_start_timestamp'] = op['start_timestamp']
            record['image_end_timestamp'] = op['end_timestamp']
            boxes.append(record)
    # build_box_records returns newest first within each image. Sort globally
    # before slicing so the sample really is the latest N complete boxes.
    boxes.sort(key=lambda b: parse_timestamp(b['end_timestamp']))
    selected = boxes[-count:]
    positive = [b for b in selected if b['box_prediction_time_s'] > 0]
    local = sum(b['box_local_explanation_time_s'] for b in selected)
    full = sum(b['box_full_pcx_explanation_time_s'] for b in selected)
    for index, box in enumerate(selected, 1):
        box['window_index'] = index
        box['local_vs_prediction_ratio'] = (box['box_local_explanation_time_s'] / box['box_prediction_time_s']) if box['box_prediction_time_s'] > 0 else None
    summary = dict(scope='box', requested_boxes=count, available_complete_boxes=len(boxes), selected_boxes=len(selected),
        status='complete' if len(selected)==count else 'insufficient_complete_boxes',
        local_explanation_vs_prediction_time=mean(b['box_local_explanation_time_s']/b['box_prediction_time_s'] for b in positive) if positive else None,
        global_explanation_vs_all_local_explanations=full/local if local else None,
        mean_box_prediction_time_s=mean(b['box_prediction_time_s'] for b in selected) if selected else None,
        mean_box_backward_time_s=mean(b['box_local_explanation_time_s'] for b in selected) if selected else None,
        mean_box_full_attribution_time_s=mean(b['box_full_pcx_explanation_time_s'] for b in selected) if selected else None,
        sum_backward_time_s=local, sum_full_attribution_time_s=full,
        valid_prediction_denominators=len(positive), zero_prediction_denominators=len(selected)-len(positive),
        timing_lines=sum(b['num_timing_lines'] for b in selected),
        distinct_images=len({b['image_index_in_fetched_logs'] for b in selected}),
        first_box_start=selected[0]['start_timestamp'] if selected else None,
        last_box_end=selected[-1]['end_timestamp'] if selected else None,
        formulas=dict(local='mean_i(sum_j Backward_ij / sum_j Prediction_ij)',
                      global_vs_local='sum_ij FullAttribution_ij / sum_ij Backward_ij'),
        notes=['Only explicitly ended boxes from object detection operations are included.',
               'Timing lines inside each box are summed before taking the mean of box ratios.',
               'Full attribution already includes backward work. No durations are added together.',
               'No standalone image Forward pass time, dataset multiplier or fixed allowed limit enters these ratios.',
               'Coordinates are not inferred: only the box/tile/detection IDs actually present in logs are exported.'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'original_box_formula_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'original_box_formula_records.json').write_text(json.dumps(selected,indent=2)+'\n')
    with (output/'original_box_formula_records.csv').open('w',newline='') as handle:
        fields=list(selected[0]) if selected else ['window_index','object_id','start_timestamp','end_timestamp']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(selected)
    return summary, selected
