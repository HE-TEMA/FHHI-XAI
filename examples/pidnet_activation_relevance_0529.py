"""Compare feature activation with predicted-flood relevance for the BRK image."""
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from LCRP.utils.crp import CondAttributionSegmentation
from LCRP.utils.pidnet_canonizers import PIDNetCanonizer, EpsilonPlusFlatforPIDNet


def main():
    logging.getLogger().setLevel(logging.WARNING)
    torch.set_num_threads(4)
    package = types.ModuleType('diagnostic_pidnet')
    package.__path__ = [str(ROOT / 'LCRP/models')]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location('diagnostic_pidnet.pidnet', ROOT / 'LCRP/models/pidnet.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    state = torch.load(ROOT / 'models/flood_model_brk2.pt', map_location='cpu')
    state = state.get('state_dict', state)
    state = {key.removeprefix('module.').removeprefix('model.'): value for key, value in state.items()}
    model = module.PIDNet(planes=32, num_classes=2, augment=True).eval()
    model.load_state_dict(state, strict=True)
    model.augment = False
    rgb = cv2.imread(str(ROOT / 'DJI_20240405181930_0529.jpg'))[:, :, ::-1]
    normalized = rgb.astype(np.float32) / 255
    normalized -= [.485, .456, .406]
    normalized /= [.229, .224, .225]
    normalized = cv2.resize(normalized, (1280, 720))
    x = torch.from_numpy(normalized.transpose(2, 0, 1).copy())[None]
    layer = 'layer5.0.conv1'
    features = {}
    handle = model.get_submodule(layer).register_forward_hook(
        lambda m, i, o: features.update(activation=o.detach().clone()))
    with torch.no_grad():
        plain = model(x)
    handle.remove()
    activation = features['activation'][0]
    # Same spatial maximum criterion used for activation-mode concept ranking.
    values, indices = activation.flatten(1).max(1).values.topk(3)
    attr = CondAttributionSegmentation(model)
    composite = EpsilonPlusFlatforPIDNet(canonizers=[PIDNetCanonizer()])
    result = attr(x.clone().requires_grad_(), [{'y': 1}], composite,
                  record_layer=[layer], init_rel=1)
    canonized = result.prediction.detach()
    heatmap = result.heatmap.detach().cpu().squeeze().numpy()
    summary = {
        'checkpoint': 'models/flood_model_brk2.pt', 'input_shape': list(x.shape),
        'layer': layer, 'activation_channel_ids': indices.tolist(),
        'activation_maxima': values.tolist(),
        'plain_predicted_flood_pixels': int((plain.argmax(1) == 1).sum()),
        'canonized_predicted_flood_pixels': int((canonized.argmax(1) == 1).sum()),
        'output_pixel_count': int(plain.shape[-2] * plain.shape[-1]),
        'canonizer_max_logit_difference': float((plain - canonized).abs().max()),
        'flood_relevance_max_abs': float(np.abs(heatmap).max()),
        'flood_relevance_finite': bool(np.isfinite(heatmap).all()),
        'relevance_target': 'class 1, masked to pixels whose argmax is class 1',
    }
    out = ROOT / 'output/brk2_pidnet_0529_xai_size'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'activation_relevance_check.json').write_text(json.dumps(summary, indent=2))
    np.save(out / 'flood_relevance.npy', heatmap)
    np.save(out / 'top_activations.npy', activation[indices].numpy())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Your image')
    axes[0, 1].imshow(plain.argmax(1)[0].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f"Plain flood mask: {summary['plain_predicted_flood_pixels']} pixels")
    limit = summary['flood_relevance_max_abs'] or 1.0
    axes[0, 2].imshow(heatmap, cmap='bwr', vmin=-limit, vmax=limit)
    axes[0, 2].set_title(f"Flood relevance: max |R| = {summary['flood_relevance_max_abs']:.3g}")
    for axis, channel, value in zip(axes[1], indices, values):
        visual = axis.imshow(activation[channel].numpy(), cmap='magma')
        axis.set_title(f'Feature channel {int(channel)}: max = {float(value):.3g}')
        fig.colorbar(visual, ax=axis, shrink=.65)
    for axis in axes.flat:
        axis.axis('off')
    fig.suptitle('Active features can exist even when predicted-flood relevance is zero')
    fig.tight_layout()
    fig.savefig(out / 'activation_vs_relevance.png', dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
