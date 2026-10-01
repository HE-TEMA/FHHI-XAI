"""Plain PIDNet inference with strict weights and a complete module shape trace.

Run from any directory: python examples/pidnet_original_forward.py
Normalization follows PIDNet_brk2/tools/custom.py; default resize matches BRK XAI.
"""
import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=ROOT / 'DJI_20240405181930_0529.jpg')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/flood_model_brk2.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/brk2_pidnet_0529_xai_size')
    parser.add_argument('--native', action='store_true', help='Keep original size instead of BRK notebook 720x1280')
    args = parser.parse_args()
    torch.set_num_threads(4)
    # Isolate the original package from the repository's other "models" packages.
    import types
    package = types.ModuleType('original_pidnet_models')
    package.__path__ = [str(ROOT / 'PIDNet_brk2/models')]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        'original_pidnet_models.pidnet', ROOT / 'PIDNet_brk2/models/pidnet.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = state.get('state_dict', state)
    state = {key.removeprefix('module.').removeprefix('model.'): value
             for key, value in state.items()}
    planes = state['conv1.0.weight'].shape[0]
    classes = state['final_layer.conv2.weight'].shape[0]
    augment = 'seghead_p.conv2.weight' in state
    model = module.PIDNet(m=2, n=3, planes=planes, num_classes=classes,
                         ppm_planes=96, head_planes=128, augment=augment)
    model.load_state_dict(state, strict=True)
    model.augment = False
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    image = bgr.astype(np.float32)[:, :, ::-1] / 255.0
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    if not args.native:
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    trace = []
    handles = []
    def shapes(value):
        if isinstance(value, torch.Tensor):
            return list(value.shape)
        return [shapes(v) for v in value]
    def hook(name):
        def record(layer, inputs, output):
            trace.append({'module': name, 'type': type(layer).__name__,
                          'input': shapes(inputs), 'output': shapes(output)})
        return record
    for name, layer in model.named_modules():
        if name:
            handles.append(layer.register_forward_hook(hook(name)))
    print(f'Strictly loaded {len(state)} tensors from {args.checkpoint}', flush=True)
    print(f'Running input {list(x.shape)} on {device}', flush=True)
    with torch.inference_mode():
        logits = model(x)
        full_logits = F.interpolate(logits, size=bgr.shape[:2], mode='bilinear', align_corners=True)
        mask = full_logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        probability = full_logits.softmax(1)[0, 1].cpu().numpy()
    for handle in handles:
        handle.remove()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / 'mask.png'), mask)
    cv2.imwrite(str(args.output / 'mask_visible.png'), mask * 255)
    overlay = bgr.copy()
    overlay[mask == 1] = (0.55 * bgr[mask == 1] + 0.45 * np.array([255, 160, 0])).astype(np.uint8)
    cv2.imwrite(str(args.output / 'overlay.jpg'), overlay)
    np.save(args.output / 'raw_logits.npy', logits.cpu().numpy())
    np.save(args.output / 'flood_probability.npy', probability)
    with (args.output / 'layer_trace.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['module', 'type', 'input', 'output'])
        writer.writeheader()
        writer.writerows(trace)
    metadata = {'image': str(args.image.resolve()), 'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'model_source': str(ROOT / 'PIDNet_brk2/models/pidnet.py'),
                'device': device, 'loaded_tensors': len(state), 'strict_load': True,
                'input_shape': list(x.shape), 'logits_shape': list(logits.shape),
                'flood_fraction': float((mask == 1).mean()),
                'classes': {'0': 'background', '1': 'flood'},
                'preprocessing': 'BGR to RGB, float32 /255, ImageNet mean/std; ' + ('no resize' if args.native else 'resize normalized input to 720x1280 as in BRK XAI notebook'),
                'output_interpolation': 'bilinear to original image size, align_corners=True',
                'internal_interpolation': 'bilinear, align_corners=False',
                'xai_hooks': False}
    (args.output / 'run.json').write_text(json.dumps(metadata, indent=2))
    (args.output / 'forward_source.py').write_text(inspect.getsource(module.PIDNet.forward))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, visual, title in zip(axes, [bgr[:, :, ::-1], mask, overlay[:, :, ::-1]],
                                  ['Original image', 'PIDNet: white = flood', 'Flood overlay']):
        axis.imshow(visual, cmap='gray', vmin=0, vmax=1) if visual.ndim == 2 else axis.imshow(visual)
        axis.set_title(title)
        axis.axis('off')
    fig.tight_layout()
    fig.savefig(args.output / 'prediction.png', dpi=130)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
