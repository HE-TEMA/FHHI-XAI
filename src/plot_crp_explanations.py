import os
import sys
import gc
import time

import copy
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.image import imgify
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes, make_grid
import zennit.image as zimage
import logging

logger = logging.getLogger(__name__)

from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from LCRP.utils.render import vis_opaque_img_border


def plot_explanations(model_name, model, dataset, sample_id, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir):
    # This code is for results visualization, taken from L-CRP/experiments/plot_crp_explanation.py
    img, t = dataset[sample_id]

    print(img.shape)

    # Get explanation visualization
    fig = plot_one_image_explanation(model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir)
    
    # Display the figure
    plt.figure(fig)
    plt.show()
    print("Done plotting.")


def log_memory(message="", reset_max=False):
    """Log memory usage in a readable format"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if reset_max:
            torch.cuda.reset_max_memory_allocated()
        print(f"MEMORY {message}: {allocated:.2f}GB (max: {max_allocated:.2f}GB)")
    else:
        print(f"MEMORY {message}: CUDA not available")


def plot_one_image_explanation_optimized(model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    log_memory("STARTING")
    
    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": class_id}]

    img = img[None, ...].to(device)
    ratio = img.shape[-2] / img.shape[-1]

    # Create the figure with axes
    fig, axs = plt.subplots(n_concepts, 3,
                           figsize=((1.6 + 1 / ratio) * n_refimgs / 4, 1.6 * n_concepts),
                           gridspec_kw={'width_ratios': [4, 4, n_refimgs * ratio]},
                           dpi=200)
    
    log_memory("AFTER FIGURE CREATION")
    
    # Process by explanation type
    if "deeplab" in model_name or "unet" in model_name:
        log_memory("BEFORE SEGMENTATION ATTR")
        
        # Make a copy to avoid modifying original
        img_copy = copy.deepcopy(img).requires_grad_()
        attr = attribution(img_copy, condition, composite, record_layer=[layer], init_rel=1)
        
        log_memory("AFTER SEGMENTATION ATTR")
        
        # Get heatmap
        heatmap = zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True)
        
        # Process mask based on output directory
        if "DLR" in output_dir:
            mask = attr.prediction[0][0]
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = mask > 0.5
        else:
            mask = (attr.prediction[0].argmax(dim=0) == class_id).detach().cpu()
        
        # Apply mask to image
        sample_ = dataset.reverse_augmentation(img[0] + 0)
        if sample_.shape[0] == 3:
            img_ = F.to_pil_image(draw_segmentation_masks(sample_, masks=mask, alpha=0.5, colors=["red"]))
        else:
            img_ = F.to_pil_image(draw_segmentation_masks(sample_[:3, :, :], masks=mask, alpha=0.5, colors=["red"]))
        
        axs[0][0].imshow(np.asarray(img_))
        axs[0][0].contour(mask.cpu().numpy(), colors="black", linewidths=[2])
        
        # Clear variables no longer needed
        del img_copy, mask, sample_
        gc.collect()
        
    elif "yolo" in model_name or "ssd" in model_name:
        log_memory("BEFORE DETECTION ATTR")
        
        attribution.take_prediction = prediction_num
        attr = attribution(img.requires_grad_(), condition, composite, record_layer=[layer], init_rel=1)
        
        log_memory("AFTER DETECTION ATTR")
        
        heatmap = np.array(zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True))
        heatmap = zimage.imgify(heatmap, symmetric=True)
        
        # Process predicted boxes
        predicted_boxes = model.predict_with_boxes(img)[1][0]
        predicted_classes = attr.prediction.argmax(dim=2)[0]
        print("Predicted classes: ", torch.unique(predicted_classes).detach().cpu().numpy())
        sorted = attr.prediction.max(dim=2)[0].argsort(descending=True)[0]
        predicted_classes = predicted_classes[sorted]
        predicted_boxes = predicted_boxes[sorted]
        predicted_boxes = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id][prediction_num]
        boxes = torch.tensor(predicted_boxes, dtype=torch.float)[None]
        colors = ["#ffcc00" for _ in boxes]
        
        result = draw_bounding_boxes((dataset.reverse_normalization(img[0])).type(torch.uint8),
            boxes, colors=colors, width=8
        )
        
        img_ = F.to_pil_image(result)
        attribution.take_prediction = 0
        
        axs[0][0].imshow(np.asarray(img_))
        
        # Clear variables
        del predicted_boxes, predicted_classes, sorted, boxes, result
        gc.collect()
        
    else:
        raise NameError
    
    log_memory("AFTER DETECTION/SEGMENTATION")
    
    # Get layer map and visualization
    layer_map = get_layer_names(model, [torch.nn.Conv2d])
    fv = VISUALIZATIONS[model_name](attribution, dataset, layer_map, preprocess_fn=lambda x: x,
                                  path=output_dir,
                                  max_target="max", device=device)
    
    # Get channel relevance
    if mode == "relevance":
        channel_rels = ChannelConcept().attribute(attr.relevances[layer], abs_norm=True)
    else:
        channel_rels = attr.activations[layer].detach().cpu().flatten(start_dim=2).max(2)[0]
        channel_rels = channel_rels / channel_rels.abs().sum(1)[:, None]
    
    # Get top concepts
    topk = torch.topk(channel_rels[0], n_concepts)
    topk_ind = topk.indices.detach().cpu().numpy()
    topk_rel = topk.values.detach().cpu().numpy()
    
    print("Concepts:", topk)
    
    attribution_start_ts = time.time()
    # Get conditional heatmaps
    conditions = [{"y": class_id, layer: c} for c in topk_ind]
    if mode == "relevance":
        log_memory("BEFORE CONDITIONAL HEATMAPS")
        attribution.take_prediction = prediction_num
        cond_heatmap, _, _, _ = attribution(img.requires_grad_(), conditions, composite, exclude_parallel=True)
        attribution.take_prediction = 0
        log_memory("AFTER CONDITIONAL HEATMAPS")
    else:
        cond_heatmap = torch.stack([attr.activations[layer][0][t] for t in topk_ind]).detach().cpu()
    
    logger.debug(f"Time to compute conditional heatmaps: {time.time() - attribution_start_ts:.2f}s")
    
    # Process reference images in small batches
    print("Computing reference images...")
    log_memory("BEFORE REF IMAGES")
    
    ref_images_start_ts = time.time()
    # Use a smaller batch size to reduce memory usage
    ref_imgs = fv.get_max_reference(topk_ind, layer, mode, (0, n_refimgs), composite=composite, rf=True,
                                  plot_fn=vis_opaque_img_border, batch_size=2)  # Reduced batch size
    
    logger.debug(f"Time to compute reference images: {time.time() - ref_images_start_ts:.2f}s")
    log_memory("AFTER REF IMAGES")
    
    # Plotting
    plotting_start_ts = time.time()
    print("Plotting...")
    resize = torchvision.transforms.Resize((150, 150))
    
    # Plot each concept in a separate loop to allow GC between concepts
    for r, row_axs in enumerate(axs):
        log_memory(f"PLOTTING CONCEPT {r}")
        
        for c, ax in enumerate(row_axs):
            if c == 0:
                if r == 0:
                    ax.set_title(f"input")
                elif r == 1:
                    ax.set_title("heatmap")
                    ax.imshow(heatmap)
                else:
                    ax.axis('off')
            
            if c == 1:
                if r == 0:
                    ax.set_title("cond. heatmap")
                ax.imshow(imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=False))
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(topk_rel[r] * 100):2.1f}%")
            
            elif c >= 2:
                if r == 0 and c == 2:
                    ax.set_title("concept visualizations")
                
                # Process reference images for this concept
                if ref_imgs and topk_ind[r] in ref_imgs:
                    # Convert and resize each reference image one at a time to save memory
                    resized_refs = []
                    for i in ref_imgs[topk_ind[r]]:
                        img_tensor = resize(torch.from_numpy(np.array(i)).permute((2, 0, 1)))
                        resized_refs.append(img_tensor)
                        
                    # Create grid and show
                    grid = make_grid(resized_refs, nrow=int(n_refimgs / 2), padding=0)
                    grid = np.array(zimage.imgify(grid.detach().cpu()))
                    ax.imshow(grid)
                    ax.yaxis.set_label_position("right")
                    
                    # Clear temp variables
                    del resized_refs, grid
                    gc.collect()
            
            ax.set_xticks([])
            ax.set_yticks([])
            
        # Clear some memory after each concept
        gc.collect()
        
    plt.tight_layout()
    
    logger.debug(f"Time to plot: {time.time() - plotting_start_ts:.2f}s")
    log_memory("AFTER PLOTTING")
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Return the figure
    return fig


# This function calls the optimized version and ensures memory is cleaned up
def plot_one_image_explanation(model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir):
    try:
        # Reset max memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()
            
        # Run optimized version
        fig = plot_one_image_explanation_optimized(model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir)
        
        # Ensure any remaining tensors are cleared
        gc.collect()
        torch.cuda.empty_cache()
        
        return fig
    
    except Exception as e:
        # In case of an error, make sure memory is cleared
        print(f"Error during explanation: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        raise


def plot_one_image_explanation_old(model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir):
    """This version of the function definitely works correctly, but for PersonVehicleDetection it takes 30GB of VRAM.
    
    To profile and optimize I introduced the optimized version above.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": class_id}]


    img = img[None, ...].to(device)
    ratio = img.shape[-2] / img.shape[-1]

    fig, axs = plt.subplots(n_concepts, 3,
                            figsize=((1.6 + 1 / ratio) * n_refimgs / 4, 1.6 * n_concepts),
                            gridspec_kw={'width_ratios': [4, 4, n_refimgs * ratio]},
                            dpi=200)

    if "deeplab" in model_name or "unet" in model_name:

        attr = attribution(copy.deepcopy(img).requires_grad_(), condition, composite, record_layer=[layer],
                           init_rel=1)
        heatmap = zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True)

        if "DLR" in output_dir:
            mask = attr.prediction[0][0]
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = mask > 0.5

        else:
            mask = (attr.prediction[0].argmax(dim=0) == class_id).detach().cpu()
        sample_ = dataset.reverse_augmentation(img[0] + 0)
        if sample_.shape[0] == 3:
            img_ = F.to_pil_image(draw_segmentation_masks(sample_, masks=mask, alpha=0.5, colors=["red"]))
        else:
            img_ = F.to_pil_image(draw_segmentation_masks(sample_[:3, :, :], masks=mask, alpha=0.5, colors=["red"]))

        axs[0][0].imshow(np.asarray(img_))
        axs[0][0].contour(mask.cpu().numpy(), colors="black", linewidths=[2])


    elif "yolo" in model_name or "ssd" in model_name:
        attribution.take_prediction = prediction_num
        attr = attribution(img.requires_grad_(), condition, composite, record_layer=[layer], init_rel=1)
        heatmap = np.array(zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True))
        heatmap = zimage.imgify(heatmap, symmetric=True)

        predicted_boxes = model.predict_with_boxes(img)[1][0]
        predicted_classes = attr.prediction.argmax(dim=2)[0]
        print("Predicted classes: ", torch.unique(predicted_classes).detach().cpu().numpy())
        sorted = attr.prediction.max(dim=2)[0].argsort(descending=True)[0]
        predicted_classes = predicted_classes[sorted]
        predicted_boxes = predicted_boxes[sorted]
        predicted_boxes = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id][prediction_num]
        boxes = torch.tensor(predicted_boxes, dtype=torch.float)[None]
        colors = ["#ffcc00" for _ in boxes]
        result = draw_bounding_boxes((dataset.reverse_normalization(img[0])).type(torch.uint8),
                                     boxes, colors=colors, width=8)

        img_ = F.to_pil_image(result)
        attribution.take_prediction = 0

        axs[0][0].imshow(np.asarray(img_))
    else:
        raise NameError

    layer_map = get_layer_names(model, [torch.nn.Conv2d])
    fv = VISUALIZATIONS[model_name](attribution, dataset, layer_map, preprocess_fn=lambda x: x,
                                    path=output_dir,
                                    max_target="max", device=device)

    if mode == "relevance":
        channel_rels = ChannelConcept().attribute(attr.relevances[layer], abs_norm=True)
    else:
        channel_rels = attr.activations[layer].detach().cpu().flatten(start_dim=2).max(2)[0]
        channel_rels = channel_rels / channel_rels.abs().sum(1)[:, None]

    topk = torch.topk(channel_rels[0], n_concepts)
    topk_ind = topk.indices.detach().cpu().numpy()
    topk_rel = topk.values.detach().cpu().numpy()

    print("Concepts:", topk)

    conditions = [{"y": class_id, layer: c} for c in topk_ind]
    if mode == "relevance":
        attribution.take_prediction = prediction_num
        cond_heatmap, _, _, _ = attribution(img.requires_grad_(), conditions, composite, exclude_parallel=True)
        attribution.take_prediction = 0
    else:
        cond_heatmap = torch.stack([attr.activations[layer][0][t] for t in topk_ind]).detach().cpu()

    print("Computing reference images...")
    ref_imgs = fv.get_max_reference(topk_ind, layer, mode, (0, n_refimgs), composite=composite, rf=True,
                                    plot_fn=vis_opaque_img_border, batch_size=4)

    print("Plotting...")
    resize = torchvision.transforms.Resize((150, 150))

    for r, row_axs in enumerate(axs):

        for c, ax in enumerate(row_axs):
            if c == 0:
                if r == 0:
                    ax.set_title(f"input")
                elif r == 1:
                    ax.set_title("heatmap")
                    ax.imshow(heatmap)
                else:
                    ax.axis('off')

            if c == 1:
                if r == 0:
                    ax.set_title("cond. heatmap")
                ax.imshow(imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=False))
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(topk_rel[r] * 100):2.1f}%")

            elif c >= 2:
                if r == 0 and c == 2:
                    ax.set_title("concept visualizations")
                grid = make_grid(
                    [resize(torch.from_numpy(np.array(i)).permute((2, 0, 1))) for i in ref_imgs[topk_ind[r]]],
                    nrow=int(n_refimgs / 2),
                    padding=0)
                grid = np.array(zimage.imgify(grid.detach().cpu()))
                ax.imshow(grid)
                ax.yaxis.set_label_position("right")

            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()

    # Return the figure instead of showing it
    return fig


# Add a new helper function to convert the figure to an array if needed
def fig_to_array(fig):
    """
    Convert a matplotlib figure to a numpy array.
    
    Args:
        fig: matplotlib figure
        
    Returns:
        numpy array of the figure
    """
    # Draw the figure to a canvas
    fig.canvas.draw()
    
    # Convert canvas to numpy array
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data