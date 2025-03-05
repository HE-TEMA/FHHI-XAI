import os
import sys
# This is needed for imports inside the LCRP submodule to work correctly
sys.path.append(os.path.join(os.path.dirname(__file__), 'LCRP'))

if __name__=="__main__":
    sys.path.append(os.getcwd())

import torch
from torchvision import transforms
import argparse
import yaml

from crp.concepts import ChannelConcept
from crp.helper import get_layer_names

from datasets.person_car_dataset import PersonCarDataset 
from canonizers import YoloV6Canonizer
from LCRP.utils.zennit_composites import EpsilonPlusFlat, EpsilonGammaFlat
from LCRP.utils.crp import CondAttributionLocalization, FeatureVisualizationLocalization
from LCRP.models.yolov6 import get_yolov6s6

# The last 2 imports will probably need to be implemented by copying from source code and changing these functions as needed:
# For FeatureVisualizationMultiTarget:
# - get_data_sample
# - Possibly get_max_reference to do some hacky implementations as i did with iToBoS
# For CondAttributionLocalization: relevance_init to initialize the relevance on the outputs that you want to explain

def main(model, dataset, output_dir, batch_size, device):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    composite = EpsilonGammaFlat(canonizers=[YoloV6Canonizer()])

    model = model.to(device)
    model.eval()

    all_layer_names = get_layer_names(model, [torch.nn.Conv2d])
    def filter_layer_condition(name):
        cond = True
        cond = cond and (name in ["backbone.ERBlock_6.2.cspsppf.cv7.block.conv"])
    # cond=True means analyse all layers.
    # CHANGE THIS FUNCTION TO RETURN True ONLY ON (AND ON ALL) LAYER NAMES THAT YOU WANT TO ANALYSE
    # something like
        #cond = False
        #cond = cond or (name in ['module.model.5.conv','module.model.7.conv'])
        #cond = cond or ("model.15.m.2" in name)
        return cond

    layer_names=[]
    for name in all_layer_names:
        if filter_layer_condition(name):
            layer_names.append(name)
    
    print("Analyzing layers: ", layer_names)

    channel_concept = ChannelConcept()
    layer_map = {layer: channel_concept for layer in layer_names}

    attribution = CondAttributionLocalization(model)

    # Make sure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # fv = FeatureVisualizationLocalizationYoloV6S6(attribution,
    fv = FeatureVisualizationLocalization(attribution,
                                    dataset,
                                    layer_map,
                                    preprocess_fn=lambda x: x,
                                    path=output_dir,
                                    max_target="max")

    fv.run(composite, 0, len(dataset), batch_size, checkpoint=100)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    dtype = torch.float32
    # Load model
    model_path = "models/best_v6s6_ckpt.pt"
    model = get_yolov6s6(ckpt_path=model_path, device=device, dtype=dtype)
    stride = int(model.stride.max())

    # Int or None
    max_detection = None

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Resize((1280, 1280)),
        transforms.Lambda(lambda x: x.to(dtype)), 
    ])


    # Load dataset
    root_dir = "datasets/data/person_car_detection_data/Arthal/"

    img_root = os.path.join(root_dir, "images")
    label_root = os.path.join(root_dir, "labels")

    # Create the dataset using the provided image and label directories
    print("Loading dataset...")

    dataset = PersonCarDataset(
        root_dir=root_dir,
        split="val",
        transform=transform,
    )
    print(f"Dataset loaded with {len(dataset)} images.")



    output_dir = "yolo_analysis_out"
    batch_size = 2

    main(model, dataset, output_dir, batch_size, device)