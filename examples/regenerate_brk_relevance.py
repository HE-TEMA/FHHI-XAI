"""Regenerate BRK CRP/PCX banks and relevance figures using the selected checkpoint."""
import importlib.util
import hashlib
import json
import logging
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from src.datasets.flood_dataset_crp import FloodDataset
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, COMPOSITES, VISUALIZATIONS


def main():
    torch.set_num_threads(4)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger('LCRP.utils.crp').setLevel(logging.CRITICAL)
    package = types.ModuleType('brk_relevance_pidnet')
    package.__path__ = [str(ROOT / 'LCRP/models')]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + '.pidnet', ROOT / 'LCRP/models/pidnet.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    state = torch.load(ROOT / 'models/flood_model_brk2.pt', map_location='cpu')
    state = state.get('state_dict', state)
    state = {k.removeprefix('module.').removeprefix('model.'): v for k, v in state.items()}
    model = module.PIDNet(planes=32, num_classes=2, augment=True).eval()
    model.load_state_dict(state, strict=True)
    model.augment = False
    dataset = FloodDataset(root_dir=str(ROOT / 'data/BRK/flood_segmentation'),
                           split='train', min_flood_coverage=.10)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    layers = [name for name in get_layer_names(model, [torch.nn.Conv2d])
              if not name.startswith(('seghead_p.', 'seghead_d.'))]
    crp = ROOT / 'examples/output/crp/CRP_BRK2_RELEVANCE'
    pcx = ROOT / 'examples/output/pcx/PCX-BRK2-RELEVANCE'
    figures = ROOT / 'examples/output/relevance_figures/BRK2'
    figures.mkdir(parents=True, exist_ok=True)
    metadata = {'checkpoint': 'models/flood_model_brk2.pt', 'mode': 'relevance',
                'checkpoint_sha256': hashlib.sha256((ROOT / 'models/flood_model_brk2.pt').read_bytes()).hexdigest(),
                'class_id': 1, 'min_ground_truth_flood_coverage': .10,
                'dataset_size': len(dataset), 'layers': layers, 'device': device,
                'samples': [str(item['img']) for item in dataset.files], 'status': 'running'}
    (figures / 'run.json').write_text(json.dumps(metadata, indent=2))
    print(f'Regenerating {len(dataset)} samples, {len(layers)} inference layers on {device}', flush=True)
    # Record flood relevance for every inference layer in one dataset pass.
    # Both banks share the exact same attribution and sample ordering.
    model.to(device)
    concept = ChannelConcept()
    attribution = ATTRIBUTORS['pidnet'](model)
    composite = COMPOSITES['pidnet'](canonizers=[CANONIZERS['pidnet']()])
    fv = VISUALIZATIONS['pidnet'](attribution, dataset, {name: concept for name in layers},
                                preprocess_fn=lambda x: x, path=str(crp), max_target='max', device=device)
    for bank in [fv.RelMax, fv.ActMax, fv.RelStats, fv.ActStats]:
        bank.SAMPLE_SIZE = 100
    fv.saved_checkpoints = {'r_max': [], 'a_max': [], 'r_stats': [], 'a_stats': []}
    banks = {name: {'activations': [], 'attributions': []} for name in layers}
    flood_pixels = []
    last_checkpoint = 0
    for index in range(len(dataset)):
        image = dataset[index][0].unsqueeze(0).to(device).requires_grad_()
        result = attribution(image, [{'y': 1}], composite, record_layer=layers, init_rel=1)
        flood_pixels.append(int((result.prediction.argmax(1) == 1).sum()))
        for name in layers:
            activation = result.activations.get(name)
            relevance = result.relevances.get(name)
            if activation is None or relevance is None:
                continue
            fv.analyze_activation(activation, name, concept, np.array([index]), np.array([1]))
            fv.analyze_relevance(relevance, name, concept, np.array([index]), np.array([1]))
            banks[name]['activations'].append(activation.amax((-2, -1)).detach().cpu().numpy())
            banks[name]['attributions'].append(concept.attribute(relevance, abs_norm=True).detach().cpu().numpy())
        del result, image
        if (index + 1) % 20 == 0:
            print(f'CRP + PCX: {index + 1}/{len(dataset)} samples', flush=True)
        if (index + 1) % 100 == 0 or index + 1 == len(dataset):
            fv._save_results((last_checkpoint, index + 1))
            last_checkpoint = index + 1
    fv.collect_results(fv.saved_checkpoints)
    metadata['predicted_flood_pixels'] = flood_pixels
    metadata['unsupported_layers'] = []
    for name, arrays in banks.items():
        folder = pcx / name
        folder.mkdir(parents=True, exist_ok=True)
        if len(arrays['attributions']) != len(dataset):
            metadata['unsupported_layers'].append(name)
            (folder / 'unsupported_layer.txt').write_text('Layer was not recorded for every sample.\n')
            continue
        with h5py.File(folder / 'pcx_bank.h5', 'w') as bank:
            bank.attrs['dataset_size'] = len(dataset)
            bank.attrs['class_id'] = 1
            bank.attrs['mode'] = 'relevance'
            for key, rows in arrays.items():
                matrix = np.concatenate(rows).astype(np.float32)
                if not np.isfinite(matrix).all():
                    raise RuntimeError(f'Nonfinite {key} for {name}')
                np.save(folder / f'{key}.npy', matrix)
                bank.create_dataset(key, data=matrix, maxshape=(None, matrix.shape[1]), chunks=True)
    metadata['status'] = 'banks_complete'
    (figures / 'run.json').write_text(json.dumps(metadata, indent=2))
    from src.plot_crp_explanations import plot_one_image_explanation
    from src.plot_pcx_pidnet_new import plot_pcx_explanations
    for sample_id, layer in [(1, 'layer5.0.conv1'), (100, 'final_layer.conv1')]:
        fig = plot_one_image_explanation('pidnet', model, dataset[sample_id][0], dataset,
                                        1, layer, 0, 'relevance', 3, 3, str(crp))
        fig.savefig(figures / f'crp_{sample_id}_{layer}.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
        fig = plot_pcx_explanations('pidnet', model, dataset, sample_id, layer, 3, 3, 5,
                                   str(ROOT / 'examples/new-ref-img-BRK-RELEVANCE'), str(crp), str(pcx))
        fig.savefig(figures / f'pcx_{sample_id}_{layer}.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
    metadata['status'] = 'complete'
    (figures / 'run.json').write_text(json.dumps(metadata, indent=2))
    print('Completed relevance regeneration:', figures, flush=True)


if __name__ == '__main__':
    main()
