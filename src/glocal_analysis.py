import math
import numpy as np

from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
import torch
from tqdm import tqdm

from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES


def _extend_pidnet_canonized_layer_names(layer_names):
    extra = [
        "final_layer.sequential.conv1",
        "final_layer.sequential.conv2",
        "seghead_p.sequential.conv1",
        "seghead_p.sequential.conv2",
        "seghead_d.sequential.conv1",
        "seghead_d.sequential.conv2",
    ]
    out = list(layer_names)
    for name in extra:
        if name not in out:
            out.append(name)
    return out


def _chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _broadcast_targets_for_feature_visualization(fv, data_batch, targets_samples, samples_batch):
    if isinstance(targets_samples, np.ndarray):
        normalized_targets = targets_samples.tolist()
        # Wrap a single sample's target vector like the list branch below does.
        if (
            len(samples_batch) == 1
            and normalized_targets
            and not isinstance(normalized_targets[0], (list, tuple, np.ndarray))
        ):
            normalized_targets = [normalized_targets]
    elif isinstance(targets_samples, (list, tuple)):
        if len(samples_batch) == 1 and targets_samples and not isinstance(targets_samples[0], (list, tuple, np.ndarray)):
            normalized_targets = [list(targets_samples)]
        else:
            normalized_targets = list(targets_samples)
    else:
        normalized_targets = [targets_samples]

    data_broadcast, targets, sample_indices = [], [], []

    try:
        for i_t, target in enumerate(normalized_targets):
            single_targets = fv.multitarget_to_single(target)
            for st in single_targets:
                targets.append(int(st))
                data_broadcast.append(data_batch[i_t])
                sample_indices.append(int(samples_batch[i_t]))
    except NotImplementedError:
        return data_batch, np.array(normalized_targets), np.array(samples_batch)

    if not data_broadcast:
        return None, None, None

    return torch.stack(data_broadcast, dim=0), np.array(targets), np.array(sample_indices)


def _run_analysis_with_recorded_layers(
    fv,
    composite,
    layer_names,
    dataset_len,
    batch_size=1,
    checkpoint=100,
    layer_chunk_size=4,
):
    print("[run_analysis] using explicit record_layer attribution path.")
    fv.saved_checkpoints = {"r_max": [], "a_max": [], "r_stats": [], "a_stats": []}

    batches = max(1, math.ceil(dataset_len / batch_size))
    last_checkpoint = 0
    last_processed_index = None
    pbar = tqdm(total=batches, dynamic_ncols=True)

    for b in range(batches):
        pbar.update(1)
        start = b * batch_size
        stop = min((b + 1) * batch_size, dataset_len)
        samples_batch = np.arange(start, stop)

        data_batch, targets_samples = fv.get_data_concurrently(samples_batch, preprocessing=True)
        data_broadcast, targets, sample_indices = _broadcast_targets_for_feature_visualization(
            fv, data_batch, targets_samples, samples_batch
        )
        if data_broadcast is None:
            continue

        conditions = [{fv.attribution.MODEL_OUTPUT_NAME: [int(t)]} for t in targets]

        for layer_chunk in _chunked(layer_names, layer_chunk_size):
            result = fv.attribution(
                data_broadcast,
                conditions,
                composite,
                record_layer=layer_chunk,
                exclude_parallel=False,
            )

            for layer_name in layer_chunk:
                concept = fv.layer_map[layer_name]
                activation = result.activations.get(layer_name)
                if activation is not None:
                    fv.analyze_activation(activation, layer_name, concept, sample_indices, targets)

                relevance = result.relevances.get(layer_name)
                if relevance is not None:
                    fv.analyze_relevance(relevance, layer_name, concept, sample_indices, targets)

            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        last_processed_index = int(sample_indices[-1])
        if b % checkpoint == checkpoint - 1:
            fv._save_results((last_checkpoint, last_processed_index + 1))
            last_checkpoint = last_processed_index + 1

    pbar.close()

    if last_processed_index is None:
        raise ValueError("CRP analysis did not process any valid targets from the dataset.")

    fv._save_results((last_checkpoint, last_processed_index + 1))
    return fv.collect_results(fv.saved_checkpoints)


def run_analysis(
    model_name,
    model,
    dataset,
    output_dir,
    device,
    class_id=1,
    use_canonizer=True,
    record_layers=None,
    layer_chunk_size=1,
):
    canonizers = [CANONIZERS[model_name]()] if use_canonizer else []
    composite = COMPOSITES[model_name](canonizers=canonizers)
    if not use_canonizer:
        print("[run_analysis] running without canonizer (requested).")

    dataset_len = len(dataset)
    if dataset_len == 0:
        raise ValueError("run_analysis received an empty dataset; no samples are available for CRP analysis.")

    model = model.to(device)
    model.eval()
    cc = ChannelConcept()
    available_layer_names = get_layer_names(model, [torch.nn.Conv2d])
    if record_layers is None:
        layer_names = available_layer_names
    else:
        layer_names = list(dict.fromkeys(record_layers))
        missing_layers = [
            layer for layer in layer_names
            if layer not in available_layer_names
        ]
        if missing_layers:
            raise ValueError(
                "Requested CRP record layer(s) do not exist: "
                f"{missing_layers}. Available convolution layers include: "
                f"{available_layer_names[:20]}"
            )
        if not layer_names:
            raise ValueError("record_layers must contain at least one layer.")
        print(
            "[run_analysis] restricting CRP to "
            f"{len(layer_names)} layer(s): {layer_names}"
        )
    if model_name == "pidnet" and use_canonizer:
        # Older PIDNet wrappers exposed ``*.sequential.*`` aliases. Do not add
        # aliases that are absent from the current model: CRP would warn and
        # perform extra forward/backward passes that can never produce banks.
        layer_names = [
            layer for layer in _extend_pidnet_canonized_layer_names(layer_names)
            if layer in available_layer_names
        ]

    attribution = ATTRIBUTORS[model_name](model)
    layer_map = {layer: cc for layer in layer_names}

    fv = VISUALIZATIONS[model_name](
        attribution,
        dataset,
        layer_map,
        preprocess_fn=lambda x: x,
        path=output_dir,
        max_target="max",
    )

    new_sample_size = 100
    fv.RelMax.SAMPLE_SIZE = new_sample_size
    fv.ActMax.SAMPLE_SIZE = new_sample_size
    fv.RelStats.SAMPLE_SIZE = new_sample_size
    fv.ActStats.SAMPLE_SIZE = new_sample_size

    return _run_analysis_with_recorded_layers(
        fv,
        composite,
        layer_names,
        dataset_len,
        batch_size=1,
        checkpoint=100,
        layer_chunk_size=layer_chunk_size,
    )
