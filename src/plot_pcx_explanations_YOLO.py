import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
# Add the parent directory to the Python path - bad practice, but it's just for the example
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append("..")
#from src.minio_client import MinIOClient
from crp.helper import get_layer_names
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from crp.concepts import ChannelConcept
from sklearn.mixture import GaussianMixture
from LCRP.utils.render import vis_opaque_img_border
from crp.image import imgify
from torchvision.utils import draw_bounding_boxes, make_grid
import torchvision.transforms.functional as F

from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
import joblib
from PIL import ImageDraw
from src.pcx_helper import get_ref_images

def plot_pcx_explanations(
    class_id, model_name, model, dataset, sample_id, n_concepts, n_refimgs, num_prototypes, prediction_num, layer_name,
        ref_imgs_path, output_dir_pcx, output_dir_crp, use_half=False):
    img, t = dataset[sample_id]

    fig = plot_one_image_pcx_explanation(
        model_name, model, img, dataset, class_id, n_concepts, n_refimgs, num_prototypes, prediction_num, layer_name,
        ref_imgs_path, output_dir_pcx, output_dir_crp, use_half=use_half)
    plt.figure(fig)

    plt.tight_layout()

    plot_dir = "output/pcx/pcx_plots"
    os.makedirs(plot_dir, exist_ok=True)

    safe_layer = layer_name.replace('.', '_')
    fname = (
        f"pcx_class{class_id}"
        f"_layer{safe_layer}"
        f"_sample{sample_id}"
        f"_n_prot{num_prototypes}"
        f"_nconc{n_concepts}.png"
    )
    fullpath = os.path.join(plot_dir, fname)

    # save plot
    fig.savefig(fullpath, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_one_image_pcx_explanation(
    model_name, model, img, dataset, class_id, n_concepts, n_refimgs, num_prototypes, prediction_num, layer_name,
        ref_imgs_path, output_dir_pcx, output_dir_crp, use_half=False):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Model has to be in eval state
    model.to(device)
    model.eval()

    if use_half:
        model.half()

    layer_names = get_layer_names(model, types=[torch.nn.Conv2d])
    num_prototypes = num_prototypes[class_id]

    # Setting up CRP
    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": class_id}]

    fv = VISUALIZATIONS[model_name](attribution,
                                    dataset,
                                    layer_names,
                                    preprocess_fn=lambda x: x,
                                    path=output_dir_crp,
                                    max_target="max")
    cc = ChannelConcept()

    # Getting the sample we selected
    data = img
    data = data[None, ...].to(device)

    # Loading relevances for this layer
    folder = f"{output_dir_pcx}/{layer_name}/"
    attributions = torch.from_numpy(np.load(folder + f"attributions_{class_id}.npy"))

    # Training GMM based on relevances if not done already
    # Initialize Gaussian Mixture Model (GMM) with specified number of prototypes as components
    cache_path = f'output/pcx/gmms/gmm_cache_{layer_name}_class_{class_id}.pkl'
    prototype_cache_path = f'output/pcx/gmm_prototypes/prototype_gmms_cache_{layer_name}_class_{class_id}.pkl'

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    os.makedirs(os.path.dirname(prototype_cache_path), exist_ok=True)

    if os.path.exists(cache_path) and os.path.exists(prototype_cache_path):
        # Load the GMM and individual GMMs from the cache files
        gmm = joblib.load(cache_path)
        prototype_gmms = joblib.load(prototype_cache_path)
    else:
        # Fit the GMM
        gmm = GaussianMixture(n_components=num_prototypes, reg_covar=1e-5, random_state=0).fit(attributions)
        # Create individual GMMs for each prototype and store them in a list
        prototype_gmms = [GaussianMixture(n_components=1, covariance_type='full', ) for p in range(num_prototypes)]
        # Save the GMM and individual GMMs to cache files
        joblib.dump(gmm, cache_path)
        joblib.dump(prototype_gmms, prototype_cache_path)

    for p, g_ in enumerate(prototype_gmms):
        g_._set_parameters([
            param[p:p + 1] if j > 0 else param[p:p + 1] * 0 + 1
            for j, param in enumerate(gmm._get_parameters())])

    # Calculating scores of the dataset, used further for outlier detection
    scores = gmm.score_samples(attributions)

    if use_half:
        data = data.half().to(device).requires_grad_(True)
    else:
        data = data.to(device).requires_grad_(True)

    # Running attribution on the input image
    attribution.take_prediction = prediction_num
    logger.debug(f"Running attribution on the input image, {attribution.take_prediction}") 
    attr = attribution(
            data,
            condition,
            composite,
            record_layer=[layer_name],
            init_rel=1)

    # Channel (neuron) relevance on the given layer for this image
    channel_rels = cc.attribute(attr.relevances[layer_name], abs_norm=True)

    # Finding how well given sample fits to prototypes, finding the closest prototype
    score_sample = gmm.score_samples(channel_rels.detach().cpu())
    likelihoods = [g_.score_samples(channel_rels.detach().cpu()) for g_ in prototype_gmms]
    mean = gmm.means_[np.argmax(likelihoods)]
    mean = torch.from_numpy(mean)
    closest_sample_to_mean = ((attributions - mean[None])).pow(2).sum(dim=1).argmin().item()

    # Closest prototype
    data_p, target_p = dataset[closest_sample_to_mean]
    if use_half:
        data_p = data_p[None, ...].to(device).half()
    else:
        data_p = data_p[None, ...].to(device)

    # Getting top concepts/neurons for the given image in the given layer
    topk = torch.topk(channel_rels[0], n_concepts)
    topk_ind = topk.indices.detach().cpu().numpy()

    # Getting reference images for those concepts
    ref_imgs = get_ref_images(fv, topk_ind, layer_name, composite=composite, class_id=class_id, n_ref=n_refimgs,
                              ref_imgs_save_path=ref_imgs_path)

    # This part is supposed to calculate conditional heatmaps and prototype heatmaps
    conditions = [{"y": class_id, layer_name: c} for c in topk_ind]

    attribution.take_prediction = prediction_num
    cond_heatmap, _, _, _ = attribution(data.requires_grad_(), conditions, composite, exclude_parallel=True)
    logger.debug(f"Running conditional attribution on the input image, {attribution.take_prediction}")

    # ─── define cache dir & files ────────────────────────────────
    cache_dir = os.path.join(output_dir_pcx, "cache", layer_name, f"class_{class_id}_protos_{num_prototypes}")
    os.makedirs(cache_dir, exist_ok=True)
    heatmap_cache = os.path.join(cache_dir, f"attr_p_heatmap_protos{num_prototypes}.npy")
    cond_cache = os.path.join(cache_dir, f"cond_heatmap_p_protos{num_prototypes}.npy")

    # ─── load or compute & cache raw heatmap arrays ────────────────
    if os.path.exists(heatmap_cache) and os.path.exists(cond_cache):
        # load back into torch
        logger.debug("Loading prototype heatmaps from cache")
        attr_p_heatmap = torch.from_numpy(np.load(heatmap_cache))
        cond_heatmap_p = torch.from_numpy(np.load(cond_cache))
        logger.debug("Loaded prototype heatmaps from cache")
    else:
        logger.debug("Cache not found, computing fresh")
        # compute them fresh
        attribution.take_prediction = 0
        cond_heatmap_p, _, _, _ = attribution(
            data_p.requires_grad_(),
            conditions,
            composite,
            exclude_parallel=True
        )
        attribution.take_prediction = 0
        attr_p = attribution(
            data_p.requires_grad_(),
            condition,
            composite,
            record_layer=[layer_name],
            init_rel=1
        )

        # detach → CPU → numpy
        heatmap_p_tensor      = attr_p.heatmap.detach().cpu()
        cond_heatmap_p_tensor = cond_heatmap_p.detach().cpu()

        # save as .npy
        np.save(heatmap_cache,      heatmap_p_tensor.numpy())
        np.save(cond_cache,         cond_heatmap_p_tensor.numpy())
        attr_p_heatmap = heatmap_p_tensor

        print("Saved prototype heatmaps to cache")


    # This was here previously
    # predicted_boxes = model.predict_with_boxes(data)[1][0]
    # Rewriting for clarity
    _, batch_predicted_boxes = model.predict_with_boxes(data)
    sample_predicted_boxes = batch_predicted_boxes[0]

    # This is already predicted as class_id
    predicted_boxes = sample_predicted_boxes[prediction_num]

    # predicted_classes = attr.prediction.argmax(dim=2)[0]
    # sorted = attr.prediction.max(dim=2)[0].argsort(descending=True)[0]
    # predicted_classes = predicted_classes[sorted]
    # predicted_boxes = predicted_boxes[sorted]
    # # Filter boxes for the d esired class.
    # filtered_boxes = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id]

    # try:
    #     predicted_boxes = filtered_boxes[prediction_num]
    # except IndexError:
    #     print(f"Warning: No bounding box found for class {class_id} at index {prediction_num}.")
    #     raise IndexError(f"No bounding box found for class {class_id} at index {prediction_num}.")

    boxes = predicted_boxes.clone().detach().float()[None]
    colors = ["#ffcc00" for _ in boxes]
    result = draw_bounding_boxes((dataset.reverse_normalization(data[0])).type(torch.uint8),
                                 boxes, colors=colors, width=8)

    img_ = F.to_pil_image(result)

    # Get bounding box coordinates.
    box_coords = predicted_boxes.clone().detach().cpu().numpy()
    x_min, y_min, x_max, y_max = box_coords.astype(int)
    orig_img_tensor = dataset.reverse_normalization(data[0])
    orig_img_pil = F.to_pil_image(orig_img_tensor.type(torch.uint8))

    # Zoom out by adding a margin
    orig_width, orig_height = orig_img_pil.size
    box_width = x_max - x_min
    box_height = y_max - y_min
    # choose zoom factor based on class
    zoom_factor = 0.2 if class_id == 1 else 2.0
    # compute margin
    margin_x = int(zoom_factor * box_width)
    margin_y = int(zoom_factor * box_height)
    crop_x_min = max(0, x_min - margin_x)
    crop_y_min = max(0, y_min - margin_y)
    crop_x_max = min(orig_width, x_max + margin_x)
    crop_y_max = min(orig_height, y_max + margin_y)

    # Crop the detection region with extra context.
    cropped_img = orig_img_pil.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
    # Draw the original bounding box (adjusted to the cropped image coordinates) with a thinner outline.
    draw = ImageDraw.Draw(cropped_img)
    adjusted_box = (x_min - crop_x_min, y_min - crop_y_min, x_max - crop_x_min, y_max - crop_y_min)
    draw.rectangle(adjusted_box, outline="yellow", width=1)

    # This was here previously
    # predicted_boxes = model.predict_with_boxes(data_p)[1][0]
    # Rewriting for clarity
    _, batch_predicted_boxes = model.predict_with_boxes(data_p)
    sample_predicted_boxes = batch_predicted_boxes[0]
    predicted_boxes = sample_predicted_boxes[0]

    # predicted_classes = attr_p.prediction.argmax(dim=2)[0]
    # print(f"Predicted boxes: {predicted_boxes}")
    # print(f"Predicted boxes shape: {predicted_boxes.shape}")

    # sorted = attr_p.prediction.max(dim=2)[0].argsort(descending=True)[0]
    # predicted_classes = predicted_classes[sorted]
    # predicted_boxes = predicted_boxes[sorted]
    # # Filter boxes for the d esired class.
    # filtered_boxes = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id]
    # predicted_boxes = filtered_boxes[prediction_num]

    boxes = predicted_boxes.clone().detach().float()[None]
    colors = ["#ffcc00" for _ in boxes]
    result = draw_bounding_boxes((dataset.reverse_normalization(data_p[0])).type(torch.uint8),
                                 boxes, colors=colors, width=8)

    img_prototype = F.to_pil_image(result)

    # Get bounding box coordinates.
    box_coords = predicted_boxes.clone().detach().cpu().numpy()
    x_min, y_min, x_max, y_max = box_coords.astype(int)
    orig_img_tensor = dataset.reverse_normalization(data_p[0])
    orig_img_pil = F.to_pil_image(orig_img_tensor.type(torch.uint8))

    # Zoom out by adding a margin
    orig_width, orig_height = orig_img_pil.size
    box_width = x_max - x_min
    box_height = y_max - y_min
    # choose zoom factor based on class
    zoom_factor = 0.2 if class_id == 1 else 2.0
    # compute margin
    margin_x = int(zoom_factor * box_width)
    margin_y = int(zoom_factor * box_height)
    crop_x_min = max(0, x_min - margin_x)
    crop_y_min = max(0, y_min - margin_y)
    crop_x_max = min(orig_width, x_max + margin_x)
    crop_y_max = min(orig_height, y_max + margin_y)

    # Crop the detection region with extra context.
    cropped_img_prot = orig_img_pil.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
    # Draw the original bounding box (adjusted to the cropped image coordinates) with a thinner outline.
    draw = ImageDraw.Draw(cropped_img_prot)
    adjusted_box = (x_min - crop_x_min, y_min - crop_y_min, x_max - crop_x_min, y_max - crop_y_min)
    draw.rectangle(adjusted_box, outline="yellow", width=1)

    # --- Defining plot ---
    width_ratios = [1, 1, n_refimgs/4, 1, 1, 1]
    n_rows = max(n_concepts, 4)  # always at least 4 rows
    fig, axs = plt.subplots(
        n_rows, 6,
        figsize=(4 * n_refimgs / 4, 1.8 * n_rows),
        gridspec_kw={'width_ratios': width_ratios},
        dpi=200
    )
    resize = torchvision.transforms.Resize((150, 150))

    for r, row_axs in enumerate(axs):
        for c, ax in enumerate(row_axs):

            if r >= n_concepts:
                if not (r == 3 and c == 0):
                    ax.axis("off")
                    continue

            # --- col 0: input / heatmap / detection / histogram ---
            if c == 0:
                if r == 0:
                    ax.set_title("input")
                    ax.imshow(img_)
                elif r == 1:
                    ax.set_title("heatmap")
                    ax.imshow(imgify(attr.heatmap.detach().cpu(),
                                     cmap="bwr", symmetric=True, level=3))
                elif r == 2:
                    ax.set_title("detection")
                    ax.imshow(cropped_img)
                elif r == 3:
                    ax.set_title("class likelihood")
                    h = ax.hist(scores, bins=20, color='k')
                    ax.vlines(score_sample, 0, h[0].max(),
                              linestyle='--', linewidth=3, label="sample")
                    ax.legend()
                    ax.set_ylabel("density")
                    ax.set_xlabel("log-likelihood")
                    ax.set_xticks([]); ax.set_yticks([])
                    # outlier label
                    lt, ut = np.percentile(scores, 1), np.percentile(scores, 99)
                    txt = ("Outlier" if score_sample < lt or score_sample > ut
                           else "Ordinary")
                    bc = dict(boxstyle="round,pad=0.3",
                              edgecolor=("red" if "Outlier" in txt else "green"),
                              facecolor=("red" if "Outlier" in txt else "green"),
                              alpha=0.3, linewidth=4)
                    ax.text(0.5, -0.35, txt, transform=ax.transAxes,
                            ha="center", fontsize=10, fontweight='bold',
                            color=("red" if "Outlier" in txt else "green"),
                            bbox=bc)

            # --- col 1: input localization ---
            elif c == 1:
                ax.imshow(imgify(cond_heatmap[r],
                                 symmetric=True, cmap="bwr", padding=True, level=3))
                ax.set_ylabel(f"concept {topk_ind[r]}\n"
                              f"relevance: {channel_rels[0,topk_ind[r]]*100:2.1f}")
                if r == 0:
                    ax.set_title("Input localization")

            # --- col 2: reference imgs grid ---
            elif c == 2:
                if r == 0:
                    ax.set_title("concept visualization")
                grid = make_grid(
                    [resize(torch.from_numpy(np.asarray(i).copy())
                             .permute(2, 0, 1)) for i in ref_imgs[topk_ind[r]]],
                    nrow=int(n_refimgs/2), padding=0
                )
                ax.imshow(grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                ax.yaxis.set_label_position("right")

            # --- col 3: ΔR boxes ---
            elif c == 3:
                if r == 0:
                    ax.set_title("Difference to prot")
                ax.imshow(np.zeros((150,150,3)), alpha=0.2)
                delta_R = ((channel_rels[0,topk_ind[r]].item()
                            - mean[topk_ind[r]].item())*100)
                if delta_R > 2.5:
                    txt, ec = f"ΔR = {delta_R:+2.1f}\n⚠ over-used", "#ff0000"
                elif delta_R < -2.5:
                    txt, ec = f"ΔR = {delta_R:+2.1f}\n⚠ under-used", "#ff0000"
                else:
                    txt, ec = f"ΔR = {delta_R:+2.1f}\n✓ similar", "#00cc00"
                rect = patches.Rectangle((0,0),150,150,
                                         linewidth=3, edgecolor=ec, facecolor="white")
                ax.add_patch(rect)
                l0, l1 = txt.split("\n")
                ax.text(75, 60, l0, ha="center", va="center",
                        bbox=dict(facecolor=ec, edgecolor="none"))
                ax.text(75, 90, l1, ha="center", va="center",
                        fontproperties=FontProperties(weight='bold'), color=ec)
                ax.axis("off")

            # --- col 4: proto localization ---
            elif c == 4:
                ax.imshow(imgify(cond_heatmap_p[r],
                                 symmetric=True, cmap="bwr", padding=True, level=3))
                ax.set_ylabel(f"concept {topk_ind[r]}\n"
                              f"relevance: {mean[topk_ind[r]]*100:2.1f}")
                if r == 0:
                    ax.set_title("Prot localization")

            # --- col 5: prototype image / heatmap / detection ---
            elif c == 5:
                if r == 0:
                    ax.set_title("prototype")
                    ax.imshow(img_prototype)
                elif r == 1:
                    ax.set_title("heatmap")
                    ax.imshow(imgify(attr_p_heatmap,
                                     cmap="bwr", symmetric=True, level=3))
                elif r == 2:
                    ax.set_title("detection")
                    ax.imshow(cropped_img_prot)
                else:
                    ax.axis("off")

            ax.set_xticks([]); ax.set_yticks([])

    return fig
