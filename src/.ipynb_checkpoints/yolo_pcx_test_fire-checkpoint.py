import os
import torch
import torchvision
import sys
import torchvision.transforms.functional as F
import matplotlib.patches as patches
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps, ImageDraw
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
import json

sys.path.append("..")
from crp.helper import get_layer_names
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from crp.concepts import ChannelConcept
from sklearn.mixture import GaussianMixture
from crp.image import imgify
from torchvision.utils import draw_bounding_boxes, make_grid
from matplotlib.font_manager import FontProperties
from src.pcx_helper_fire import get_detection_crop_input, get_ref_images, get_detection_crop, prot_with_concepts, export_gmm_view_html, get_detection_crop_exact


def find_matching_box_idx(model, data_tensor, target_box, device, iou_thresh=0.3):
    """Find detection index matching target_box by IoU."""
    data_tensor = data_tensor.to(device).requires_grad_(True)
    with torch.enable_grad():
        scores_all, boxes_all = model.predict_with_boxes(data_tensor)
    scores_all, boxes_all = scores_all.detach(), boxes_all.detach()
    boxes = boxes_all[0] if boxes_all.ndim == 3 else boxes_all
    if boxes.shape[0] == 0:
        return 0, 0.0

    if not torch.is_tensor(target_box):
        target_box = torch.tensor(target_box, dtype=torch.float32)
    target_box = target_box.to(boxes.device).flatten()[:4]

    x1 = torch.max(target_box[0], boxes[:, 0])
    y1 = torch.max(target_box[1], boxes[:, 1])
    x2 = torch.min(target_box[2], boxes[:, 2])
    y2 = torch.min(target_box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ious = inter / (area1 + area2 - inter + 1e-6)

    return int(ious.argmax()), float(ious.max())


def plot_pcx_explanations(class_id, model_name, model, img, orig_img, dataset, orig_dataset,
                          n_concepts=5, n_refimgs=12, num_prototypes=None, prediction_num=0,
                          layer_name="decoder.center.0.0",
                          ref_imgs_path="../output/ref_imgs/", output_dir_pcx="../output_synthetic/pcx/yolo_person_car",
                          output_dir_crp="../output/crp/yolo_person_car/", plot_prot_crops=True,
                          letterbox_shape=None, original_shape=None, rescale_boxes_fn=None):
    """
    Tile-based PCX explanation plotting.
    
    dataset = exhaustive_dataset (tiles, 640x640)
    orig_dataset = PersonCarDataset (full images)
    meta["dataset_idx"] -> index into exhaustive_dataset
    meta["img_idx"] -> index into PersonCarDataset (full images)
    meta["box"] -> box in tile coordinates (640x640), no rescaling needed
    """

    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # layers and prototypes
    layer_names = get_layer_names(model, types=[torch.nn.Conv2d])
    num_prototypes = num_prototypes[class_id]

    # CRP setup
    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": class_id}]
    fv = VISUALIZATIONS[model_name](attribution, dataset, layer_names,
                                    preprocess_fn=lambda x: x,
                                    path=output_dir_crp,
                                    max_target="max")
    cc = ChannelConcept()

    # get input
    data = img
    data = data[None, ...].to(device)

    # load attributions
    folder = f"{output_dir_pcx}/{layer_name}/"
    attr_path = os.path.join(folder, f"attributions_{class_id}.npy")
    meta_path = os.path.join(folder, f"meta_class_{class_id}.json")

    attributions = torch.from_numpy(np.load(attr_path))
    with open(meta_path, "r") as f:
        meta = json.load(f)

    assert attributions.shape[0] == len(meta), \
        f"per-det rows mismatch: A={attributions.shape[0]} vs meta={len(meta)}"

    # GMM fitting/loading
    cache_path = f'{output_dir_pcx}/gmms/gmm_cache_{layer_name}_class_{class_id}_prot_{num_prototypes}.pkl'
    prototype_cache_path = f'{output_dir_pcx}/gmm_prototypes/prototype_gmms_cache_{layer_name}_class_{class_id}_prot_{num_prototypes}.pkl'

    if os.path.exists(cache_path):
        gmm = joblib.load(cache_path)
    else:
        gmm = GaussianMixture(n_components=num_prototypes, reg_covar=1e-5, random_state=0).fit(attributions)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump(gmm, cache_path)

    # --- REORDER BY COVERAGE ---
    A_np = attributions.detach().cpu().numpy().astype(np.float32)
    labels_raw = gmm.predict(A_np)
    K = gmm.n_components
    counts = np.bincount(labels_raw, minlength=K).astype(float)
    order = np.argsort(-counts)
    perm_inv = np.empty_like(order)
    perm_inv[order] = np.arange(K)

    def _reorder_first_axis(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim >= 1 and arr.shape[0] == K:
            return arr[order, ...]
        return arr

    gmm.means_ = _reorder_first_axis(gmm.means_)
    if hasattr(gmm, 'weights_'):              gmm.weights_ = _reorder_first_axis(gmm.weights_)
    if hasattr(gmm, 'covariances_'):          gmm.covariances_ = _reorder_first_axis(gmm.covariances_)
    if hasattr(gmm, 'precisions_'):           gmm.precisions_ = _reorder_first_axis(gmm.precisions_)
    if hasattr(gmm, 'precisions_cholesky_'):  gmm.precisions_cholesky_ = _reorder_first_axis(gmm.precisions_cholesky_)

    order_cache_dir = os.path.join(output_dir_pcx, "gmms", "orders")
    os.makedirs(order_cache_dir, exist_ok=True)
    np.save(os.path.join(order_cache_dir, f"order_by_coverage_{layer_name}_class_{class_id}_K{K}.npy"), order)

    # --- REBUILD prototype_gmms ---
    prototype_gmms = []
    base_params = gmm._get_parameters()
    for p in range(K):
        g1 = GaussianMixture(n_components=1, covariance_type=gmm.covariance_type)
        g1._set_parameters([
            (base_params[j][p:p + 1] if j > 0 else base_params[j][p:p + 1] * 0 + 1.0)
            for j in range(len(base_params))
        ])
        prototype_gmms.append(g1)

    os.makedirs(os.path.dirname(prototype_cache_path), exist_ok=True)
    joblib.dump(prototype_gmms, prototype_cache_path)

    # dataset scores
    scores = gmm.score_samples(attributions)
    data = data.to(device).requires_grad_(True)

    # attribution on input
    attribution.take_prediction = prediction_num
    attr = attribution(
        data,
        condition,
        composite,
        record_layer=[layer_name],
        init_rel=1)

    channel_rels = cc.attribute(attr.relevances[layer_name], abs_norm=True)
    channel_rels = channel_rels.detach().cpu().float()

    # --- sample fit (mixture) ---
    score_sample = gmm.score_samples(channel_rels.detach().cpu())

    # === PREP ARRAYS & CHOOSE PROTOTYPE ===
    x_star = channel_rels.detach().cpu().numpy()
    A = attributions.detach().cpu().numpy().astype(np.float32)

    post = gmm.predict_proba(x_star)
    chosen_proto = int(post.argmax(axis=1)[0])

    # === CLASS-LEVEL PERCENTILE ===
    scores = gmm.score_samples(A)
    score_star = float(score_sample[0])
    p_mix = ((scores < score_star).sum() + 0.5) / (len(scores) + 1)

    # === COMPONENT-LOCAL PERCENTILE ===
    lbl = gmm.predict(A)
    idx_k = np.where(lbl == chosen_proto)[0]
    A_k = A[idx_k]
    g_k = prototype_gmms[chosen_proto]
    scores_k = g_k.score_samples(A_k)
    s_star_k = float(g_k.score_samples(x_star)[0])
    p_local = ((scores_k < s_star_k).sum() + 0.5) / (len(scores_k) + 1)
    coverage = len(idx_k) / max(1, len(A))

    # === MEAN & MAHALANOBIS NEAREST SAMPLE ===
    mean = torch.from_numpy(gmm.means_[chosen_proto])

    mu = gmm.means_[chosen_proto].astype(np.float32)
    L = gmm.precisions_cholesky_[chosen_proto].astype(np.float32)
    diff = A - mu[None, :]
    y = diff @ L.T
    m2 = np.sum(y * y, axis=1)
    closest_row = int(np.argmin(m2))

    # === TILE-BASED: Get prototype tile and original image ===
    tile_ds_idx = int(meta[closest_row]["dataset_idx"])  # index into exhaustive_dataset
    proto_img_idx = int(meta[closest_row].get("img_idx", 0))  # index into PersonCarDataset
    box_idx_proto = int(meta[closest_row]["box_idx"])

    print(f"\n{'=' * 60}")
    print(f"[DEBUG] PROTOTYPE SELECTION:")
    print(f"  closest_row in meta: {closest_row}")
    print(f"  tile_ds_idx (exhaustive_dataset index): {tile_ds_idx}")
    print(f"  proto_img_idx (PersonCarDataset index): {proto_img_idx}")
    print(f"  box_idx_proto: {box_idx_proto}")
    print(f"  meta entry: {meta[closest_row]}")
    print(f"{'=' * 60}\n")

    # Load prototype tile from exhaustive_dataset (already 640x640)
    data_p_tile, _ = dataset[tile_ds_idx]
    data_p = data_p_tile[None, ...].to(device).requires_grad_(True)

    # Load original full image for detection crop context
    orig_img_p_raw, _ = orig_dataset[proto_img_idx]
    if torch.is_tensor(orig_img_p_raw):
        orig_img_p_pil = F.to_pil_image(orig_img_p_raw.byte() if orig_img_p_raw.dtype == torch.uint8
                                        else (orig_img_p_raw * 255).byte())
    else:
        orig_img_p_pil = Image.fromarray(np.array(orig_img_p_raw).astype(np.uint8))

    orig_np_p = np.array(orig_img_p_pil)
    original_shape_p = orig_np_p.shape[:2]

    # === IoU matching for box_idx_proto ===
    # Boxes are in tile coordinates (640x640), use meta["box"] directly
    target_box_tile = torch.tensor(meta[closest_row]["box"], dtype=torch.float32, device=device)

    with torch.enable_grad():
        scores_p_all, boxes_p_all = model.predict_with_boxes(data_p)
    scores_p_all, boxes_p_all = scores_p_all.detach(), boxes_p_all.detach()
    scores_p = scores_p_all[0] if scores_p_all.ndim == 3 else scores_p_all
    boxes_p = boxes_p_all[0] if boxes_p_all.ndim == 3 else boxes_p_all
    n_det = boxes_p.shape[0]

    if n_det > 0:
        best_iou_v, best_idx_v = 0, 0
        for i in range(n_det):
            b = boxes_p[i]
            x1_i, y1_i = max(target_box_tile[0], b[0]), max(target_box_tile[1], b[1])
            x2_i, y2_i = min(target_box_tile[2], b[2]), min(target_box_tile[3], b[3])
            inter = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
            area1 = (target_box_tile[2] - target_box_tile[0]) * (target_box_tile[3] - target_box_tile[1])
            area2 = (b[2] - b[0]) * (b[3] - b[1])
            iou_v = float(inter / (area1 + area2 - inter + 1e-6))
            if iou_v > best_iou_v:
                best_iou_v, best_idx_v = iou_v, i
        box_idx_proto = best_idx_v
        print(f"[ATTR FIX] Using box_idx_proto={box_idx_proto} (IoU={best_iou_v:.3f}) for attribution")
    elif box_idx_proto >= n_det:
        box_idx_proto = 0

    # Draw prototype with stored box (in tile coordinates)
    box_to_draw = torch.tensor(meta[closest_row]["box"], dtype=torch.float32).unsqueeze(0)
    img_uint8_p = (data_p[0] * 255).clamp(0, 255).byte().cpu()
    result = draw_bounding_boxes(img_uint8_p, box_to_draw, colors=["#ffcc00"], width=8)
    img_prototype = F.to_pil_image(result)

    # Detection crop for prototype — use tile as source, box in tile coords
    ctx = 2.0 if class_id == 0 else 0.4
    predicted_box_p = np.array(meta[closest_row]["box"], dtype=np.float32)
    tile_pil_p = F.to_pil_image(data_p_tile)
    detection_crop_p = get_detection_crop_input(
        orig_img=tile_pil_p,
        box=predicted_box_p,
        input_size=(640, 640),
        context=ctx,
        draw_box=True,
        letterbox_mode=False,
    )

    print(f"[DEBUG] original_shape_p (computed): {original_shape_p}")
    print(f"[DEBUG] tile shape: {data_p_tile.shape}")
    print(f"[DEBUG CROP] predicted_box_p: {predicted_box_p}")

    # --- extra diagnostics ---
    gamma = post[0]
    pi_k = float(gmm.weights_[chosen_proto])

    diff_k = A_k - mu[None, :]
    y_k = diff_k @ L.T
    m2_k = np.sum(y_k * y_k, axis=1)

    y_star = (x_star - mu[None, :]) @ L.T
    m2_star = float(np.sum(y_star * y_star))
    p_m2 = ((m2_k >= m2_star).sum() + 0.5) / (len(m2_k) + 1)

    # ------ TOP-K concepts ------
    rel = channel_rels[0].detach().cpu()
    proto = mean.detach().cpu()
    combined = torch.max(rel, proto)

    init_k = n_concepts
    fill_k = 0

    init_idxs = torch.topk(rel, init_k).indices.tolist()

    delta = (channel_rels[0].detach().cpu() - mean).abs()
    cand = torch.argsort(delta, descending=True).tolist()

    proto_rank = torch.topk(proto.abs(), k=proto.numel()).indices.tolist()
    fill_idxs = [i for i in proto_rank if i not in init_idxs][:fill_k]

    topk_ind = init_idxs + fill_idxs

    # get reference images
    ref_imgs = get_ref_images(fv, topk_ind, layer_name,
                              composite=composite, class_id=class_id,
                              n_ref=n_refimgs,
                              ref_imgs_save_path=f"{ref_imgs_path}/ref_imgs_12/")

    # conditional heatmaps
    conditions = [{"y": class_id, layer_name: c} for c in topk_ind]
    attribution.take_prediction = prediction_num
    cond_heatmap, _, _, _ = attribution(data.requires_grad_(), conditions,
                                        composite, exclude_parallel=True)

    # === Prototype heatmaps ===
    attribution.take_prediction = box_idx_proto
    cond_heatmap_p, _, _, _ = attribution(
        data_p.requires_grad_(),
        conditions,
        composite,
        exclude_parallel=True
    )

    attribution.take_prediction = box_idx_proto
    attr_p = attribution(
        data_p.requires_grad_(),
        condition,
        composite,
        record_layer=[layer_name],
        init_rel=1
    )

    heatmap_p_tensor = attr_p.heatmap.detach().cpu()
    cond_heatmap_p_tensor = cond_heatmap_p.detach().cpu()

    predicted_classes_p = attr_p.prediction.argmax(dim=2)[0]
    sorted_idxs_p = attr_p.prediction.max(dim=2)[0].argsort(descending=True)[0]

    attr_p_heatmap = heatmap_p_tensor
    cond_heatmap_p = cond_heatmap_p_tensor

    # === PROTOTYPE+CONCEPT FIGURE ===
    if plot_prot_crops:

        NUM_SAMPLES_PER_PROTO = 6
        K_CONCEPTS_PER_PROTO = 3
        N_REF_PER_CONCEPT = 6

        A_np = attributions.detach().cpu().numpy()
        means = gmm.means_.astype(np.float32)
        K, C = means.shape

        # Nearest samples to each prototype mean (Mahalanobis)
        diff_all = A_np[:, None, :] - gmm.means_[None, :, :]
        L_all = getattr(gmm, "precisions_cholesky_", None)
        if L_all is not None:
            y_all = np.einsum("nkc,kdc->nkd", diff_all, L_all)
            m2_all = np.einsum("nkd,nkd->nk", y_all, y_all)
        else:
            m2_all = np.einsum("nkc,nkc->nk", diff_all, diff_all)
        ranked = np.argsort(m2_all, axis=0)

        # Build crop strips per prototype — TILE-BASED
        all_crops = [[] for _ in range(K)]
        for pj in range(K):
            taken = 0
            k = 0
            max_try = min(ranked.shape[0], NUM_SAMPLES_PER_PROTO * 50)
            while taken < NUM_SAMPLES_PER_PROTO and k < max_try:
                row_idx = int(ranked[k, pj])
                k += 1
                tile_idx_p = int(meta[row_idx]["dataset_idx"])  # exhaustive_dataset index
                bx_idx = int(meta[row_idx]["box_idx"])

                try:
                    # Load tile from exhaustive_dataset
                    tile_data, _ = dataset[tile_idx_p]
                    tile_pil = F.to_pil_image(tile_data)

                    box_p = np.asarray(meta[row_idx]["box"], dtype=np.float32)
                    crop = get_detection_crop_input(
                        orig_img=tile_pil,
                        box=box_p,
                        input_size=(640, 640),
                        context=(2.0 if class_id == 0 else 0.4),
                        draw_box=True,
                        letterbox_mode=False,
                    )
                    t = torchvision.transforms.ToTensor()(crop)
                    all_crops[pj].append(t)
                    taken += 1
                except Exception:
                    # Fallback: use the tile directly
                    tile_data_fb, _ = dataset[tile_idx_p]
                    all_crops[pj].append(tile_data_fb.clamp(0, 1))
                    taken += 1

            # pad if still short
            while len(all_crops[pj]) < NUM_SAMPLES_PER_PROTO:
                row0 = int(ranked[0, pj])
                tile0 = int(meta[row0]["dataset_idx"])
                tile_data_fb, _ = dataset[tile0]
                all_crops[pj].append(tile_data_fb.clamp(0, 1))

        # --- select top concepts across prototypes ---
        M = torch.from_numpy(means).float()
        k_pick = int(min(K_CONCEPTS_PER_PROTO, C))

        per_proto_lists = []
        for pj in range(K):
            row = M[pj]
            pos_mask = row > 0
            if torch.any(pos_mask):
                num_pos = int(pos_mask.sum().item())
                take = min(k_pick, num_pos)
                masked = row.masked_fill(~pos_mask, float('-inf'))
                _, idx = torch.topk(masked, k=take, largest=True)
            else:
                _, idx = torch.topk(row, k=k_pick, largest=False)
            per_proto_lists.append(idx.tolist())

        flat = [int(i) for lst in per_proto_lists for i in lst]
        seen = set()
        top_concepts = []
        for i in flat:
            if i not in seen:
                seen.add(i)
                top_concepts.append(i)

        top_concepts = [int(i) for i in top_concepts]
        N_CONCEPTS = len(top_concepts)

        ref_imgs_concepts = get_ref_images(
            fv, top_concepts, layer_name,
            composite=composite, class_id=class_id,
            n_ref=N_REF_PER_CONCEPT,
            ref_imgs_save_path=f"{ref_imgs_path}/ref_imgs_6/"
        )

        labels = gmm.predict(A_np)
        counts = np.bincount(labels, minlength=K).astype(float)
        coverage_pct = (counts / max(1, A_np.shape[0])) * 100.0

        mu_mean = A_np.mean(axis=0).astype(np.float32)
        mu_n = mu_mean / (np.linalg.norm(mu_mean) + 1e-12)
        Pn = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
        sim_mean = (Pn @ mu_n)

        concept_matrix = torch.from_numpy(means[:, top_concepts]).T

        THUMB = 120
        resize_square = T.Compose([T.Resize(THUMB), T.CenterCrop((THUMB, THUMB))])
        all_resized = [[resize_square(t.clamp(0, 1)) for t in col] for col in all_crops]

        top_row_ratio = max(8, NUM_SAMPLES_PER_PROTO + 2)
        fig_pc, axs_pc = plt.subplots(
            nrows=N_CONCEPTS + 1, ncols=K + 1,
            figsize=(K + 6, N_CONCEPTS + 6), dpi=170,
            gridspec_kw={'width_ratios': [6] + [1] * K, 'height_ratios': [top_row_ratio] + [1] * N_CONCEPTS}
        )

        for pj in range(K):
            grid = torchvision.utils.make_grid(all_resized[pj], nrow=1, padding=1)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            grid_np = (grid_np * 255.0).clip(0, 255).astype(np.uint8)
            axs_pc[0, pj + 1].imshow(grid_np, aspect='auto')
            axs_pc[0, pj + 1].set_title(
                f"Prototype {pj}\nCovers {coverage_pct[pj]:.0f}%\nSim. {sim_mean[pj]:.2f}", fontsize=9)
            axs_pc[0, pj + 1].axis("off")
        axs_pc[0, 0].axis("off")

        for i, cidx in enumerate(top_concepts):
            imgs = ref_imgs_concepts.get(int(cidx), [])
            tiles = []
            for im in imgs[:N_REF_PER_CONCEPT]:
                if isinstance(im, Image.Image):
                    t = F.to_tensor(im).clamp(0, 1)
                else:
                    arr = np.asarray(im)
                    if arr.ndim == 3:
                        t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0).clamp(0, 1)
                    else:
                        continue
                tiles.append(resize_square(t))

            if len(tiles) == 0:
                tiles = [torch.zeros(3, THUMB, THUMB)]

            nrow = max(1, min(len(tiles), N_REF_PER_CONCEPT))
            grid = make_grid(tiles, nrow=nrow, padding=0)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            grid_np = (grid_np * 255.0).clip(0, 255).astype(np.uint8)

            axs_pc[i + 1, 0].imshow(grid_np)
            axs_pc[i + 1, 0].set_ylabel(f"concept {int(cidx)}", rotation=90, labelpad=8)
            axs_pc[i + 1, 0].set_yticks([])
            axs_pc[i + 1, 0].set_xticks([])

        vmax = float(concept_matrix.abs().max().item())
        for i in range(N_CONCEPTS):
            for j in range(K):
                val = concept_matrix[i, j].item()
                axs_pc[i + 1, j + 1].imshow([[abs(val)]], vmin=0, vmax=vmax,
                                             cmap=("Reds" if val >= 0 else "Blues"))
                color = "white" if abs(val) > 0.5 * vmax else "black"
                axs_pc[i + 1, j + 1].text(0, 0, f"{val * 100:.1f}", ha="center", va="center", color=color, fontsize=10)
                axs_pc[i + 1, j + 1].axis("off")

        plt.tight_layout()

        plot_dir = f"../output_RAS/pcx/pcx_plots"
        out_png = os.path.join(plot_dir, f"{layer_name}_class{class_id}_K{K}_proto_concepts.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig_pc.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig_pc)
        print(f"➡️ prototype+concept figure: {out_png}")

    # === MAIN PLOT: Sample vs Prototype ===
    scores_det, boxes_det = model.predict_with_boxes(data)
    scores_det = scores_det[0].detach().cpu()  # <-- Add .cpu() here
    boxes_det = boxes_det[0].detach().cpu()    # <-- Add .cpu() here

    # No rescaling needed — boxes are in tile coords
    if letterbox_shape is not None and original_shape is not None and rescale_boxes_fn is not None:
        boxes_det_np = rescale_boxes_fn(
            boxes_det,
            letterbox_shape=letterbox_shape,
            original_shape=original_shape
        )
        if isinstance(boxes_det_np, np.ndarray):
            boxes_det = torch.from_numpy(boxes_det_np)

    scores_attr = attr.prediction[0].detach().cpu()
    A_s = scores_det / (scores_det.norm(dim=1, keepdim=True) + 1e-9)
    B_s = scores_attr / (scores_attr.norm(dim=1, keepdim=True) + 1e-9)
    sim_s = A_s @ B_s.T
    mapped = sim_s.argmax(dim=1)

    print(f"[DBG] prediction_num(draw)={prediction_num}  mapped_to_attr_idx={int(mapped[prediction_num])}")

    if prediction_num >= boxes_det.shape[0]:
        raise IndexError(f"Only {boxes_det.shape[0]} detections, asked for #{prediction_num}")

    predicted_box = boxes_det[int(prediction_num)]
    pred_confidence = float(scores_det[prediction_num, class_id].item())

    # thresholds
    MIX_THR = 0.01
    LOCAL_THR = 0.01
    M2_THR = 0.05
    SMALL_THR = 0.10
    CONF_THR = 0.50
    GAMMA_THR = 0.85

    global_outlier = (p_mix < MIX_THR)
    locally_typical = (p_local >= LOCAL_THR) or (p_m2 >= M2_THR) or (gamma[chosen_proto] >= GAMMA_THR)
    small_component = (coverage <= SMALL_THR) or (pi_k <= SMALL_THR)
    is_correct = (pred_confidence >= CONF_THR)
    show_extra_diag = global_outlier and locally_typical and small_component and is_correct

    pred_label = dataset.class_names[class_id]

    # Get box in tile coordinates for drawing
    with torch.enable_grad():
        _, boxes_raw = model.predict_with_boxes(data)
    boxes_raw = boxes_raw[0].detach()
    box_tile_coords = boxes_raw[prediction_num].clone().detach().float()[None]

    colors = ["#ffcc00"]
    # For tile-based: data is already [0,1] float, just convert to uint8
    img_uint8 = (data[0].detach() * 255).clamp(0, 255).byte().cpu()
    result = draw_bounding_boxes(img_uint8, box_tile_coords, colors=colors, width=8)
    img_ = F.to_pil_image(result)

    # Detection crop — use tile as source
    tile_pil_input = F.to_pil_image(img.cpu())
    detection_crop = get_detection_crop_input(
        orig_img=tile_pil_input,
        box=predicted_box.detach().cpu().numpy() if torch.is_tensor(predicted_box) else predicted_box,
        input_size=(640, 640),
        context=ctx,
        draw_box=True,
        letterbox_mode=False,
    )

    # Draw a red rectangle on a copy to verify box location
    from PIL import ImageDraw
    debug_img = orig_img_p_pil.copy()
    draw = ImageDraw.Draw(debug_img)
    draw.rectangle(predicted_box_p.tolist(), outline="red", width=10)

    # ---- export interactive HTML (per detection!) ----
    safe_layer = layer_name.replace('.', '_')
    print('[DBG] exporting HTML with test_predicted_box in input coords');
    export_gmm_view_html(
        attributions=attributions,
        gmm=gmm,
        dataset=dataset,
        orig_dataset=dataset,
        model=model,
        class_id=class_id,
        get_detection_crop_fn=get_detection_crop_exact,
        get_detection_crop_input_fn=get_detection_crop_input,
        meta=meta,
        test_channel_rels=channel_rels[0],
        test_orig_img=orig_img,
        test_predicted_box=predicted_box,
        test_context=ctx,
        device=device,
        input_size=(letterbox_shape[1], letterbox_shape[0]) if letterbox_shape is not None else (inW, inH),
        # USE ORIGINAL SHAPE (W, H)
        score_thresh=0.4,
        input_prediction_num=prediction_num,
        export_dir=f"../output_RAS/pcx/export_html/gmm_export_layer_{safe_layer}_class_{class_id}_det{int(prediction_num):02d}",
        max_points=2000,
        title=f"GMM 3D — {layer_name} | class {class_id} | det #{int(prediction_num)}",
        reducer="umap",
        reducer_n_components=3,
        reducer_kwargs=dict(n_neighbors=30, min_dist=0.05, metric="cosine")
    )

    from src.pcx_helper_fire import export_gmm_view_2d
    
    fig = export_gmm_view_2d(
        attributions=attributions,
        gmm=gmm,
        dataset=dataset,
        orig_dataset=orig_dataset,
        model=model,
        class_id=class_id,
        get_detection_crop_fn=get_detection_crop_exact,
        get_detection_crop_input_fn=get_detection_crop_input,
        meta=meta,
        test_channel_rels=channel_rels[0],
        test_orig_img=orig_img,
        test_predicted_box=predicted_box,
        test_context=ctx,
        device=device,
        export_dir=f"../output_RAS/pcx/gmm_2d/{layer_name}",
        title=f"GMM 2D — {layer_name} | class {class_id}",
        reducer="umap",
        show_kde=True,
    )
    plt.close(fig)

    # ------- PLOTTING -------
    width_ratios = [1, 1, n_refimgs / 4, 1, 1, 1]
    n_rows = max(n_concepts, 5 if show_extra_diag else 4)
    fig, axs = plt.subplots(
        n_rows, 6,
        figsize=(4 * n_refimgs / 4, 1.8 * n_rows),
        gridspec_kw={'width_ratios': width_ratios},
        dpi=200
    )
    resize = torchvision.transforms.Resize((150, 150))

    print(f"[LL] mixture p={p_mix:.3f} | local p={p_local:.3f} | M2 p={p_m2:.3f} | "
          f"γ={gamma[chosen_proto]:.2f} | π={pi_k:.2f} | coverage={coverage * 100:.1f}% "
          f"| show_extra_diag={show_extra_diag}")

    chan_rel_signed = cc.attribute(attr.relevances[layer_name], abs_norm=False).detach().cpu()[0]
    chan_rel_abs = cc.attribute(attr.relevances[layer_name], abs_norm=True).detach().cpu()[0]

    # Populate the subplots
    for r, row_axs in enumerate(axs):
        for c, ax in enumerate(row_axs):

            if c in (1, 2, 3, 4) and r >= n_concepts:
                ax.axis("off")
                continue

            if c == 0:
                if r == 0:
                    ax.set_title("input")
                    img_ = img_.resize((150, 150), Image.BILINEAR)
                    ax.imshow(img_)
                elif r == 1:
                    ax.set_title("heatmap")
                    img_hm = imgify(attr.heatmap.detach().cpu(), cmap="bwr", symmetric=True, level=5)
                    img_hm = img_hm.resize((150, 150), Image.BILINEAR)
                    ax.imshow(img_hm)
                elif r == 2:
                    ax.set_title("Detection", fontsize=10)
                    label_str = f"{pred_label} {pred_confidence * 100:.1f}%"
                    ax.text(0.02, 0.98, label_str, transform=ax.transAxes, fontsize=7,
                            fontweight="bold", color="yellow", va="top", ha="left",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.3, edgecolor="none"))
                    ax.imshow(detection_crop)
                elif r == 3:
                    a = ax.hist(scores, bins=30, density=True, color='k', alpha=0.85)
                    ax.vlines(score_star, 0, (a[0].max() if len(a[0]) else 1),
                              linestyle='--', linewidth=3, label="sample")
                    ax.legend()
                    ax.set_ylabel("density")
                    ax.set_xlabel("log-likelihood")
                    ax.set_yticks([])
                    ax.set_xticks([])
                    lower_threshold = np.percentile(scores, 5)
                    outlier_text = "Outlier" if score_sample < lower_threshold else "Ordinary"
                    bbox_props = dict(boxstyle="round,pad=0.3",
                                      edgecolor="red" if outlier_text == "Outlier" else "green",
                                      facecolor="red" if outlier_text == "Outlier" else "green", alpha=0.3, linewidth=4)
                    ax.text(0.5, -0.35, outlier_text, transform=ax.transAxes, ha="center", fontsize=10,
                            fontweight='bold', color="red" if outlier_text == "Outlier" else "green", bbox=bbox_props)
                else:
                    ax.axis("off")

            if c == 1:
                if r == 0:
                    ax.set_title("Input localization")
                cond_h = imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=True, level=5)
                cond_h = cond_h.resize((150, 150), Image.BILINEAR)
                ax.imshow(cond_h)
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(channel_rels[0][topk_ind[r]] * 100):2.1f}")

            elif c == 2:
                if r == 0:
                    ax.set_title("concept visualization")
                grid = make_grid(
                    [resize(torch.from_numpy(np.asarray(i).copy()).permute((2, 0, 1))) for i in ref_imgs[topk_ind[r]]],
                    nrow=int(n_refimgs / 2), padding=0)
                grid_np = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                ax.imshow(grid_np)
                ax.yaxis.set_label_position("right")

            elif c == 3:
                plt.rc('text', usetex=False)
                plt.rcParams['font.family'] = 'DejaVu Sans'
                bold_font = FontProperties(weight='bold')

                if r == 0:
                    ax.set_title("Difference to prot")
                ax.imshow(np.zeros((150, 150, 3)), alpha=0.2, cmap=None)
                delta_R = (channel_rels[0][topk_ind[r]].round(decimals=3) - mean[topk_ind[r]].round(decimals=3)) * 100
                if delta_R > 2.5:
                    textstr = f"ΔR = {delta_R:+2.1f}\n⚠ over-used"
                    edge_color = "#ff0000"
                elif delta_R < -2.5:
                    textstr = f"ΔR = {delta_R:+2.1f}\n⚠ under-used"
                    edge_color = "#ff0000"
                else:
                    textstr = f"ΔR = {delta_R:+2.1f}\n✓ similar"
                    edge_color = "#00cc00"

                rect = patches.Rectangle((0, 0), 150, 150, linewidth=3, edgecolor=edge_color, facecolor='white')
                ax.add_patch(rect)
                lines = textstr.split('\n')
                ax.text(75, 60, lines[0], fontsize=10, verticalalignment='center', horizontalalignment='center',
                        bbox=dict(facecolor=edge_color, edgecolor='none'))
                ax.text(75, 90, lines[1], fontproperties=bold_font, verticalalignment='center',
                        horizontalalignment='center', color=edge_color)
                ax.set_xlim([0, 150])
                ax.set_ylim([0, 150])
                ax.axis("off")

            elif c == 5:
                if r == 0:
                    ax.set_title("prototype")
                    img_prototype_r = img_prototype.resize((150, 150), Image.BILINEAR)
                    ax.imshow(img_prototype_r)
                elif r == 1:
                    ax.set_title("heatmap")
                    img_hm_p = imgify(attr_p_heatmap, cmap="bwr", symmetric=True, level=5)
                    img_hm_p = img_hm_p.resize((150, 150), Image.BILINEAR)
                    ax.imshow(img_hm_p)
                elif r == 2:
                    ax.set_title("detection")
                    ax.imshow(detection_crop_p)
                else:
                    ax.axis("off")
            elif c == 4:
                if r == 0:
                    ax.set_title("Prot localization")
                cond_h_p = imgify(cond_heatmap_p[r], symmetric=True, cmap="bwr", padding=True, level=5)
                cond_h_p = cond_h_p.resize((150, 150), Image.BILINEAR)
                ax.imshow(cond_h_p)
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(mean[topk_ind[r]] * 100):2.1f}")

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    return fig