# PCX-TEMA
# data : https://aischnacken.slack.com/docs/T0162BZ537W/F09S95JB9LG
# GETING THE LOGS 

curl -i -X POST http://localhost:8080/tfa02/post_data \
  -H 'Content-Type: application/json' \
  --data @tests/ImageMetadata.json
docker logs explanation_tfa02 2>&1 | tail -n 1000000




This repository contains the code for applying the PCX method for TEMA project. 

## Setting up

### Load the data and models

Data and models are available on Google Drive https://drive.google.com/drive/folders/1vmkyJzojacZUc2rz-VBw5T07KzoFslzB?usp=sharing. 

Path for data is `datasets/data`.

#### 1.PIDNet model: 
- Checkpoint: Checkpoints/flood_model.pt
- Dataset: Data/flood_segmentation.zip
- Task: specifically for this model - flood segmentation

#### 2. YOLOv6s6 model:
- Checkpoint: Checkpoints/best_v6s6_ckpt.pt
- Dataset: Data/PersonCarDetectionData 
- Task: person and car detection



### Build the Docker image

To build for the TEMA cloud.
`docker build --platform linux/amd64 -t explanation_tfa02 .`

For development on a mac build like this:
`docker build -t explanation_tfa02 .`




### Push the image to the registry

- Obtain Personal Access Token (PAT) by following [these instructions](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

    It is likely that you already have a PAT stored in your git credential helper.
    To see it, do
    ``` bash
    echo "url=https://github.com" | git credential fill
    ```

- Set the PAT environment variable to your PAT value:
     ```bash
     export PAT=<your_git_personal_access_token>
     ```

- Test GHCR login:
     ```bash
     echo $PAT | docker login ghcr.io -u <your_git_username> --password-stdin
     ```

 - Push the TFA-02 container image to the Container registry:
     ```bash
     docker tag explanation_tfa02 ghcr.io/he-tema/explanation_tfa02:1.0
     docker push ghcr.io/he-tema/explanation_tfa02:1.0
     ```
- Write to Nicola Colosi on tema slack to deploy the pushed container on the TEMA cluster.

### Run the docker container

Edit the file if you want to change some environment variables
```bash
./run_docker.sh
```

### Development

#### Install dependencies
```bash
conda create -n tema python=3.8
conda activate tema
pip install -r requirements.txt
```

During development, you can run the application components separately in different terminal tabs for easier debugging and log monitoring:

1. Start Redis server:
```bash
redis-server
```

2. Start the Flask application:
```bash
DEBUG=1 python app.py
```
DEBUG=1 will make the app reload on any code changes, remove if you don't want this.

3. Start the worker process:
```bash
python worker.py
```
For the worker to work properly on a Mac, before running it do 
```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

The application will be available at `http://localhost:8080/tfa02` by default.

Note: For production deployment, use the Docker container as described in the "Run the docker container" section above.

### Testing

You can test the application using the provided test script located in the `tests/` folder. 
```bash
cd tests
```

There are two ways to run the tests:
1. Local testing (sends notifications to local Redis):
```bash
python test_post_data.py ImageMetadata
```

2. Cloud testing (sends notifications to TEMA cloud):
```bash
python test_post_data.py ImageMetadata --cloud
```

The test script will send sample image metadata to the application and you should see the processing results in the logs of both the Flask application and the worker process.


### References

**LRP — Layer-wise Relevance Propagation**

- Bach et al., [*On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation*](https://doi.org/10.1371/journal.pone.0130140)
- Montavon et al., [*Layer-Wise Relevance Propagation: An Overview*](https://doi.org/10.1007/978-3-030-28954-6_10)
- [Zennit toolbox](https://github.com/chr5tphr/zennit)

**CRP — Concept Relevance Propagation**

- Achtibat et al., [*From Attribution Maps to Human-Understandable Explanations through Concept Relevance Propagation*](https://doi.org/10.1038/s42256-023-00711-8)
- [Zennit-CRP toolbox](https://github.com/rachtibat/zennit-crp)

**L-CRP — Concept Relevance Propagation for Localization Models**

- Dreyer et al., [*Revealing Hidden Context Bias in Segmentation and Object Detection through Concept-specific Explanations*](https://arxiv.org/pdf/2211.11426)
- [L-CRP code](https://github.com/maxdreyer/L-CRP/tree/main)

**PCX — Prototypical Concept-based Explanations**

- Dreyer et al., [*Understanding the (Extra-)Ordinary: Validating Deep Model Decisions with Prototypical Concept-based Explanations*](https://arxiv.org/pdf/2311.16681)
- [PCX code](https://github.com/maxdreyer/pcx/tree/main)

---

# Complete developer and user guide

This section extends the original project notes above. It describes the workflow implemented by the current code, especially the validated YOLOv6 person/car path used by the service.

## What this repository does

The repository combines four related explanation ideas:

1. **LRP (Layer-wise Relevance Propagation)** explains a selected model output by redistributing its prediction score backward, layer by layer, toward the input. The redistribution follows the contributions made by lower-layer neurons and is designed around relevance conservation. At input level, the resulting relevance scores show how individual pixels contributed to the selected prediction.
2. **CRP (Concept Relevance Propagation)** extends LRP by conditioning the backward relevance flow on selected hidden-layer units. In convolutional networks, this project interprets individual channels as latent concepts. CRP therefore identifies which concepts contributed to a prediction, assigns concept relevance scores, and produces concept-conditional heatmaps showing where those concepts occur. Relevance Maximization (RelMax) complements this by retrieving dataset examples for which a concept was strongly relevant to a prediction, rather than merely highly activated.
3. **L-CRP (CRP for Localization Models)** adapts this local-and-global, or **glocal**, concept analysis to object detection and semantic segmentation. For an object detector, relevance is initialized from the class score of a selected bounding box, allowing a separate explanation for each detected object. For segmentation, relevance can be initialized from the selected class region or another region of interest. The explanation communicates which latent concepts contributed, how strongly they contributed, and where they are localized in the input.
4. **PCX (Prototypical Concept-based Explanations)** represents each decision by its concept-relevance vector and models recurring class-wise decision strategies with prototypes.Gaussian mixture models group similar relevance vectors and their component means represent prototypical concept-use strategies. A local prediction can then be compared with its nearest prototype. This helps reveal which learned decision strategy was used and whether a prediction deviates from normal prototypical behavior, which can indicate an outlier, a spurious strategy, or a data-quality issue.

The **TFA-02 service** applies these methods operationally: it receives an Orion/NGSI entity, obtains the corresponding image from MinIO, queues the work through Redis/RQ, runs the detector or segmenter and its PCX explanation, stores the explanation images, and updates the external entity.

CRP and PCX are not independent:

```text
                      images + labels + checkpoint + preprocessing
                                            |
                                            v
                          validate and select usable predictions
                                            |
                                            v
                          calculate the CRP relevance statistics 
                                            |
                                            v
                      generate the class-specific attribution banks 
                                            |
                                            v
clustering GMM to compute cluster centroids (prototypes) and store the reference images for concepts
                                            |
                                            v
                generating the explanation plot for a new model prediction
```

The checkpoint, class order, dataset order, preprocessing, selected layer, CRP relevance-statistics directory, PCX attribution-bank directory, and reference-image directory form one versioned set of explanation data. Do not mix results generated by different versions of any of these inputs.

## Which files should I start with?

For the current production YOLOv6 workflow, use these files in order:

1. [`examples/yolo_BRK_preprocecing_check.ipynb`](examples/yolo_BRK_preprocecing_check.ipynb) to inspect image/label preprocessing and prediction boxes corniations and make sure that we generate the same output as AUTH.
2. [`examples/run_yolov6_crp.py`](examples/run_yolov6_crp.py) for a command-line CRP smoke test.
3. [`examples/yolo_brk_crp.ipynb`](examples/yolo_brk_crp.ipynb) to calculate one-layer production CRP relevance statistics, on the validtated and selected data.
4. [`examples/yolo_brk_pcx.ipynb`](examples/yolo_brk_pcx.ipynb) to create class-specific PCX banks, metadata, GMMs, reference images, and plots.
5. [`examples/car_prototype_audit.ipynb`](examples/car_prototype_audit.ipynb) to audit whether the learned car prototypes are meaningful.
6. [`src/explanator.py`](src/explanator.py) to deploy the code and pcx setting set used by the service.

[`examples/yolo_concept_prototype.ipynb`](examples/yolo_concept_prototype.ipynb) It is useful for exploration, but the BRK notebooks above are the production workflow.

For the current flood/PIDNet workflow, start with [`examples/pidnet.ipynb`](examples/pidnet.ipynb) for layer selection, continue with [`examples/pidnet_crp_pcx.ipynb`](examples/pidnet_crp_pcx.ipynb), and compare the BRK experiments in [`examples/pidnet_BRK.ipynb`](examples/pidnet_BRK.ipynb) and [`examples/pidnet_BRK_outlier.ipynb`](examples/pidnet_BRK_outlier.ipynb). The service ultimately calls [`src/plotpcx_gpu.py`](src/plotpcx_gpu.py).

## Complete `examples/` catalog

The `examples/` directory contains research notebooks, production-preparation notebooks, command-line helpers, and generated explanation results. Not every notebook is required for deployment.

### YOLOv6 notebooks and scripts

| File | What it contains | When to use it |
|---|---|---|
| [`yolo_BRK_preprocecing_check.ipynb`](examples/yolo_BRK_preprocecing_check.ipynb) | BRK image/label preprocessing checks, including the detector input and annotation geometry | Run before expensive CRP when data, resize, padding, or any preprocecing step have changed |
| [`run_yolov6_crp.py`](examples/run_yolov6_crp.py) | Validates YOLO labels, maps dataset classes to detector classes, reproduces 640-pixel YOLOv6 letterboxing, filters predictions by same-class IoU, constructs `DetectionSubset`, and runs CRP | Recommended CLI smoke test and reusable CRP entry point |
| [`yolo_brk_crp.ipynb`](examples/yolo_brk_crp.ipynb) | Full validated BRK CRP workflow, selection diagnostics, one-layer CRP, result checks, | Authoritative notebook for the production YOLO CRP relevance statistics and selection manifest |
| [`extract_crp_manifest.py`](examples/extract_crp_manifest.py) | Recovers the exact sample/class selection from saved output in an older CRP notebook and writes a manifest | Migration/recovery tool only; new runs should write the manifest directly |
| [`yolo_brk_pcx.ipynb`](examples/yolo_brk_pcx.ipynb) | Loads the CRP manifest, exports per-class attribution banks and metadata, fits/caches GMM prototypes, displays every prototype, and generates PCX plots | Authoritative notebook for production YOLO PCX |
| [`yolo_concept_prototype.ipynb`](examples/yolo_concept_prototype.ipynb) | Concept/prototype exploration, verify its constants before reusing outputs |
| [`car_prototype_audit.ipynb`](examples/car_prototype_audit.ipynb) | Focused visual and quantitative audit of car prototypes | Use after fitting PCX to select or validate the car prototype count |

### PIDNet/flood notebooks and scripts

| File | What it contains | When to use it |
|---|---|---|
| [`pidnet.ipynb`](examples/pidnet.ipynb) | PIDNet layer comparison and recommendations for flood concept analysis | Select a meaningful explanation layer before regenerating CRP/PCX |
| [`pidnet_BRK.ipynb`](examples/pidnet_BRK.ipynb) | BRK-specific PIDNet concept/prototype experiments | BRK flood analysis and prototype tuning |
| [`pidnet_BRK_outlier.ipynb`](examples/pidnet_BRK_outlier.ipynb) | BRK prototype fitting and outlier-score experiments, including single-image/debug paths | Investigate unusual samples and prototype coverage |




## Complete `src/` catalog

### Runtime orchestration, infrastructure, and shared utilities

| File | Responsibility |
|---|---|
| [`src/__init__.py`](src/__init__.py) | Performs package initialization, applies compatibility fixes for NumPy, handles optional `wandb` imports, and exposes LCRP models. |
| [`src/explanator.py`](src/explanator.py) | Central runtime orchestrator: lazy-loads models/datasets, selects the handler for each entity type, runs YOLO or PIDNet PCX, handles CUDA memory fallbacks, records KPIs, and constructs explanation results |
| [`src/entities.py`](src/entities.py) | NGSI-LD templates and builders for person/vehicle and flood explanation entities |
| [`src/minio_client.py`](src/minio_client.py) | MinIO image/text upload and download wrapper plus bucket constants |
| [`src/kpi_logging.py`](src/kpi_logging.py) | GPU-aware timing, structured KPI record creation, log paths, JSON-line persistence, and rolling average windows |
| [`src/memory_logging.py`](src/memory_logging.py) | Small CUDA memory logging helper |
| [`src/device_utils.py`](src/device_utils.py) | Parses and validates CPU/CUDA device specifications and resolves a consistent `torch.device` |
| [`src/letterbox_utils.py`](src/letterbox_utils.py) | YOLOv6-compatible letterbox preprocessing, training preprocessing wrapper, image-size checks, and mapping boxes back to the original image |
| [`src/yolo_class_mapping.py`](src/yolo_class_mapping.py) | Enforces detector class-name order and class-safe detection/prototype box matching |
| [`src/utils_DLR.py`](src/utils_DLR.py) | NumPy rolling-window, tiling, and untile/blending helpers for large DLR imagery |
| [`src/utils/utils.py`](src/utils/utils.py) | PIDNet/general training utilities: full-model loss wrapper, metrics, logging, confusion matrix, and learning-rate adjustment |

`src/minio_client.py` currently defines the MinIO connection constants in source code.

### CRP and PCX core

| File | Responsibility | Status |
|---|---|---|
| [`src/glocal_analysis.py`](src/glocal_analysis.py) | Dataset-wide CRP analysis, multi-target broadcasting, explicit layer recording, layer chunking, checkpoint saves, and CRP statistics | Current shared CRP implementation |
| [`src/plot_crp_explanations.py`](src/plot_crp_explanations.py) | Local CRP plotting, heatmap rendering,CRP plot functions | Current CRP visualization used by `Explanator` |
| [`src/pcx_helper.py`](src/pcx_helper.py) | Shared YOLO PCX helpers for class-specific reference caching, detection image crops, prototype sample grids, and interactive GMM HTML/2-D views | Current helper for YOLO PCX |
| [`src/plot_pcx_explanations_YOLO.py`](src/plot_pcx_explanations_YOLO.py) | Current class-locked YOLO PCX pipeline: validates attribution banks/metadata/GMM caches, attributes on dataset images, chooses the same-class prototype, loads references, and renders the explanation | Current production YOLO implementation |
| [`src/plotpcx_gpu.py`](src/plotpcx_gpu.py) | Device-aware PIDNet PCX pipeline with reference retrieval, precision controls, heatmap alignment, memory reduction, CPU fallback, plotting, and outlier scoring | Current production PIDNet implementation |
| [`src/plot_pcx_proto_concept_matrix.py`](src/plot_pcx_proto_concept_matrix.py) | Builds prototype-versus-concept matrices and prototype-only figures | Analysis/visualization helper |
| [`src/yolo_pcx_test.py`](src/yolo_pcx_test.py) | Earlier standalone YOLO PCX plotting and target-box matching experiment | Experimental; not called by the service and only used in notebooks |

### Alternative and legacy PCX implementations

These files are retained for research history and notebook compatibility and usage. They have overlapping names but are not the service entry points:

| File | Scope |
|---|---|
| [`src/plot_pcx_all.py`](src/plot_pcx_all.py) | Earlier general/PIDNet PCX plotter and outlier scoring |
| [`src/plot_pcx_explanations.py`](src/plot_pcx_explanations.py) | Earlier UNet-oriented PCX implementation |
| [`src/plot_pcx_explanations_old.py`](src/plot_pcx_explanations_old.py) | Explicit older copy of the UNet PCX implementation |
| [`src/plot_pcx_explanations_PIDNET.py`](src/plot_pcx_explanations_PIDNET.py) | Earlier PIDNet-specific plotting implementation |
| [`src/plot_pcx_pidnet_new.py`](src/plot_pcx_pidnet_new.py) | Newer research variant with layer/path resolution and diverse concept/reference selection |
| [`src/plot_pcx_pidnet_integration.py`](src/plot_pcx_pidnet_integration.py) | Integration-oriented PIDNet variant with defensive sample extraction and message panels |

When changing deployed behavior, begin at the imports in `src/explanator.py`: it currently imports YOLO from `plot_pcx_explanations_YOLO.py` and PIDNet from `plotpcx_gpu.py`. Editing one of the other variants will not change the service unless its import is deliberately switched and tested.

### Dataset modules

| File | Responsibility | Status |
|---|---|---|
| [`src/datasets/base_dataset.py`](src/datasets/base_dataset.py) | Base segmentation dataset preprocessing, resizing/cropping, normalization, label handling, and augmentation utilities | Shared base |
| [`src/datasets/person_car_dataset.py`](src/datasets/person_car_dataset.py) | Naturally sorted paired image/YOLO-label reader with fixed `("person", "car")` classes | Current YOLO dataset |
| [`src/datasets/detection_subset.py`](src/datasets/detection_subset.py) | Manifest-backed view that maps CRP indices to original dataset indices and exposes validated detector-class targets | Current YOLO CRP/PCX consistency layer |
| [`src/datasets/flood_dataset.py`](src/datasets/flood_dataset.py) | Scans/pairs RGB flood images and masks, supports list-driven ordering, mask suffix normalization, and configurable segmentation preprocessing | Current flood dataset used by the service |
| [`src/datasets/General_Flood_v3.py`](src/datasets/General_Flood_v3.py) | Original/list-driven General Flood v3 dataset implementation | Training/research compatibility |
| [`src/datasets/DLR_dataset.py`](src/datasets/DLR_dataset.py) | Dataset adapter for DLR imagery | DLR research path |
| [`src/datasets/visualization.py`](src/datasets/visualization.py) | Draws dataset objects and optional padded regions | Dataset inspection helper |
| [`src/datasets/flood_dataset_crp.py`](src/datasets/flood_dataset_crp.py) | Flood dataset variant created for CRP experiments | Experimental/legacy |
| [`src/datasets/flood_dataset_metrics.py`](src/datasets/flood_dataset_metrics.py) | Flood dataset variant used in metric experiments | Experimental/legacy |
| [`src/datasets/flood_test.py`](src/datasets/flood_test.py) | Flood dataset test/experiment variant | Experimental/legacy |
| [`src/datasets/__init__.py`](src/datasets/__init__.py) | Marks the dataset package | Package marker |

Because several flood modules define a class named `FloodDataset`, always check the import rather than relying on the class name. Production uses `from src.datasets.flood_dataset import FloodDataset`.

### Developer tools

| File | Responsibility |
|---|---|
| [`src/tools/visualize_ignore_regions.py`](src/tools/visualize_ignore_regions.py) | Command-line visualization of ignored segmentation regions and sample geometry |

Generated `src/__pycache__/` files are interpreter caches, not source files, and should not be edited.

### Files outside `examples/` and `src/` that control the flow

| Path | Purpose |
|---|---|
| [`LCRP/models/`](LCRP/models) | Model wrappers and checkpoint loading |
| [`LCRP/utils/crp_configs.py`](LCRP/utils/crp_configs.py) | Model-specific attributors, canonizers, composites, and feature visualizations |
| [`app.py`](app.py) | Flask/NGSI API and task enqueueing |
| [`tasks.py`](tasks.py) | MinIO download, queued explanation execution, result upload, and status updates |
| [`worker.py`](worker.py) | RQ worker process |
| [`common_app_funcs.py`](common_app_funcs.py) | Redis connection/state, queue, entity update, and task-status helpers |
| [`Dockerfile`](Dockerfile), [`supervisord.conf`](supervisord.conf), [`run_docker.sh`](run_docker.sh) | Container build and process deployment |
| [`tests/`](tests) | Unit tests and external-service integration helpers |

### Generated files under `output/`

| Path | Meaning |
|---|---|
| `new-ref-img-BRK/*.h5` | Cached PIDNet concept reference images, one HDF5 file per layer |
| `ref_images_pidnet_flood_BRK/*.h5` | BRK flood reference-image cache |
| `test_prediction_DJI_0234.png` | Generated example prediction image |
| `__pycache__/` | Python bytecode cache; not an input and safe to regenerate |
| `.DS_Store` | macOS filesystem metadata; not used by the project |

Do not treat an HDF5 reference cache as portable across checkpoints, datasets, preprocessing, or layers. Prefer creating versioned caches under `output/` for new production runs.
## Installation

The pinned environment was built for Python 3.8:

```bash
conda create -n tema python=3.8
conda activate tema
pip install -r requirements.txt
```

For notebook work, also install Jupyter and the analysis packages if they are not already installed transitively:

```bash
pip install jupyter scikit-learn
jupyter lab
```

Run all commands from the repository root so imports such as `src` and `LCRP` resolve consistently.

### Expected input layout

The validated YOLO path expects paired images and YOLO text labels:

```text
data/BRK/person_vehicle_detection/
├── images/
│   └── train/
│       ├── image_001.jpg
│       └── ...
└── labels/
    └── train/
        ├── image_001.txt
        └── ...
```

Each label row is:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`. The current class contract is `0=person`, `1=car`; the code accepts the display synonym `vehicle` only where it explicitly normalizes names. Image and label stems must match.

The flood path expects:

```text
data/General_Flood_v3/
├── RGB/train/JPEG/
└── annotations/train/JPEG/
```

The `FloodDataset` can also receive a root that already points at `General_Flood_v3`. Mask naming/pairing behavior, crop size, normalization, augmentation, and optional list-file ordering are defined in `src/datasets/flood_dataset.py`.

## Quick start: calculate YOLOv6 CRP

First place the two-class checkpoint at `models/best_ckpt.pt`, or pass a different path explicitly. Start with a small scan:

```bash
python examples/run_yolov6_crp.py \
  --checkpoint models/best_ckpt.pt \
  --dataset-root data/BRK/person_vehicle_detection \
  --output-dir output/crp/yolo_person_car_smoke \
  --device cuda:0 \
  --scan-limit 100 \
  --iou-threshold 0.5
```

If that succeeds, run the complete dataset by removing `--scan-limit` and use a new output directory:

```bash
python examples/run_yolov6_crp.py \
  --checkpoint models/best_ckpt.pt \
  --dataset-root data/BRK/person_vehicle_detection \
  --output-dir output/crp/yolo_person_car_full \
  --device cuda:0 \
  --iou-threshold 0.5
```

The script performs the following safeguards before CRP:

1. validates image/label pairing and label syntax;
2. reproduces the YOLOv6 640-pixel letterbox geometry;
3. runs the detector and keeps rank-0, same-class predictions that match ground truth at the requested IoU;
4. creates a `DetectionSubset`, because empty post-NMS results cannot provide a differentiable target;
5. runs glocal CRP and writes relevance/activation maxima and statistics.

For production CRP results, follow the final one-layer section of `examples/yolo_brk_crp.ipynb`. It records only:

```text
module.backbone.ERBlock_3.0.rbr_dense.conv
```

and writes `crp_filter_manifest.json`. Restricting CRP to the layer used by PCX is substantially cheaper and ensures that the runtime can reconstruct the exact filtered dataset. The generic CLI currently records all convolution layers; use the notebook or pass `record_layers=[...]` when calling `run_analysis(...)` directly if only the production layer is required.

Expected CRP files include directories such as:

```text
output/crp/<run>/
├── crp_filter_manifest.json
├── RelMax_sum_normed/
├── RelMax_max_normed/
├── ActMax_max_normed/
├── RelStats_sum_normed/
├── RelStats_max_normed/
└── ActStats_max_normed/
```

Before continuing, confirm the selected layer has `_data.npy`, `_rel.npy`, and `_rf.npy` under the relevant CRP result directory.

## Next: calculate YOLOv6 PCX

Open `examples/yolo_brk_pcx.ipynb` and run it top to bottom after changing its path/configuration cell to the new:

- checkpoint;
- original dataset root;
- CRP output directory containing the manifest;
- fresh PCX output directory;
- fresh reference-image directory;
- `CRP_LAYER`.

The notebook:

1. loads and validates `crp_filter_manifest.json`;
2. rebuilds the exact `DetectionSubset` used by CRP;
3. calculates a concept-relevance vector for every validated image/class target at `CRP_LAYER`;
4. stores `attributions_0.npy`, `attributions_1.npy`, `meta_class_0.json`, and `meta_class_1.json`;
5. fits one GMM per class;
6. finds representative samples and concept reference images;
7. calls `src/plot_pcx_explanations_YOLO.py` to verify a complete explanation.

The output layout used by the service is:

```text
output/pcx/<run>/
├── <layer>/
│   ├── attributions_0.npy
│   ├── attributions_1.npy
│   ├── meta_class_0.json
│   └── meta_class_1.json
├── gmms/
├── gmm_prototypes/
├── cache_v2/
└── pcx_plots/
```

Keep each `.npy` bank and its matching metadata JSON together. The plotting code rejects length mismatches, cross-class prototype use, missing class detections, incompatible GMM feature dimensions, and a requested prototype count that differs from a cached model.

## Flood/PIDNet CRP and PCX

Use `examples/pidnet_crp_pcx.ipynb` for the full research workflow and `examples/pidnet_BRK.ipynb` for the current BRK data exploration. At runtime, `src/explanator.py` and `src/plotpcx_gpu.py` use:

- target class `1` (flood);
- layer `layer5.0.conv1`;
- `3` displayed concepts;
- `12` reference images;
- `2` prototypes;
- CRP directory `output/crp/pidnet_flood/`;
- PCX directory `output/pcx/pidnet_flood/`;
- reference directory `output/ref_imgs_pidnet/`.

The current PIDNet model loader obtains its checkpoint/configuration through `LCRP/models/pidnet.py`; update that loader when replacing the segmentation weights. Ensure the dataset preprocessing used to build CRP/PCX is the same preprocessing used at inference.

## Hyperparameters and configuration

| Parameter | Current/default value | Where | Effect |
|---|---:|---|---|
| YOLO input size | `640` | `run_yolov6_crp.py`, `Explanator`, `letterbox_utils.py` | Must match training/inference geometry and stored boxes |
| YOLO stride | `32` | same | Controls letterbox padding alignment |
| CRP IoU threshold | `0.5` | `--iou-threshold` | Higher values give cleaner ground-truth matches but fewer PCX samples |
| CRP scan limit | all | `--scan-limit` | Use a small value only for smoke tests |
| CRP canonizer | enabled | `--no-canonizer` disables it | Model-specific graph/rule preparation; keep enabled for production CRP results |
| CRP recorded layers | all conv layers by CLI; one production layer in notebook | `run_analysis(record_layers=...)` | More layers cost more time/storage; PCX requires its chosen layer to be present |
| CRP batch size | `1` | `src/glocal_analysis.py` | GPU-memory/runtime tradeoff |
| CRP layer chunk size | `4` | `src/glocal_analysis.py` | Number of recorded layers per attribution pass |
| CRP save checkpoint interval | `100` batches | `src/glocal_analysis.py` | Frequency of partial result writes |
| CRP statistic sample size | `100` | `src/glocal_analysis.py` | Number of maxima retained by each CRP statistic |
| YOLO production PCX layer | `module.backbone.ERBlock_3.0.rbr_dense.conv` | `src/explanator.py` and notebooks | Defines concept-vector dimension; changing it invalidates all PCX/GMM/reference caches |
| YOLO concepts shown | `3` | `src/explanator.py` | Top relevant channels displayed per explanation |
| YOLO reference images | `12` | `src/explanator.py` | Examples displayed for each concept; higher values use more time/memory |
| YOLO prototypes | person `4`, car `5` | `prototype_dict` in `src/explanator.py` | Per-class GMM component count |
| GMM regularization | `1e-5` | `src/plot_pcx_explanations_YOLO.py` | Stabilizes covariance fitting |
| GMM random seed | `0` | same | Makes fitting reproducible |
| Minimum runtime detection confidence | `0.30` | `PERSON_VEHICLE_MIN_EXPLANATION_CONFIDENCE` | Skips expensive explanations for low-confidence boxes |
| PIDNet PCX layer | `layer5.0.conv1` | `src/explanator.py` | Flood concept layer |
| PIDNet concepts/references/prototypes | `3 / 12 / 2` | `src/explanator.py` | Flood plot and clustering complexity |

Choose `num_prototypes` only after inspecting cluster quality and sample count. It must be positive and cannot sensibly exceed the number of attribution rows for that class. Changing it requires deleting or, preferably, writing to a new GMM/cache directory. `n_concepts` cannot exceed the available channel-relevance vector length. `n_refimgs` cannot create examples that do not exist; the code may return fewer references and PIDNet may reduce the effective count under memory pressure.

## Adding new data, weights, or preprocessing

Use a new run directory whenever CRP relevance statistics, PCX attribution banks, GMM prototypes, or reference images are recalculated. This makes rollback possible and prevents stale caches from silently influencing results.

### If only new test images arrive

No retraining or CRP/PCX regeneration is required if the checkpoint, classes, preprocessing, and reference dataset are unchanged. The service can explain a new image using the existing class prototypes. Images that belong to the CRP manifest can use their stored sample identity; unseen images use the live-image path.

### If the reference/training dataset changes

1. Put the data in the layout above.
2. If layout, file naming, label encoding, or classes differ, update `src/datasets/person_car_dataset.py` or `src/datasets/flood_dataset.py`.
3. Update `class_names` and all explicit class mappings. For YOLO production, also update `resolve_display_class_names` behavior and the hard validation in `src/explanator.py`.
4. Re-run dataset validation, CRP selection, CRP, and PCX from scratch.
5. Generate a new manifest and new attribution/metadata banks.
6. Point runtime environment variables to the matching new CRP results, PCX banks, prototype models, and reference images.

Dataset order is significant: the manifest and PCX metadata contain dataset indices. Adding, removing, or renaming files can shift the natural sort order, so old manifests must not be reused.

### If model weights change

1. Verify the architecture and output class order.
2. Pass the new checkpoint to CRP and update `PERSON_VEHICLE_CHECKPOINT` for runtime.
3. Re-run selection, CRP, PCX banks, GMMs, and reference images.
4. Use new CRP/PCX/reference directories.

The runtime intentionally compares the checkpoint basename in `crp_filter_manifest.json` with the runtime checkpoint basename. Treat a mismatch as a required regeneration, not as a check to bypass.

### If preprocessing changes

For YOLO, update both offline and online paths:

- `YOLOv6TrainPreprocess(...)` in `examples/run_yolov6_crp.py` and `src/explanator.py`;
- `letterbox_transform(...)` and box rescaling behavior in `src/letterbox_utils.py`;
- the matching geometry in `_ground_truth_in_model_coordinates(...)`;
- notebook preprocessing cells used to create the manifest and PCX banks.

For PIDNet, update the dataset/base-dataset parameters and the inference transform together. Any change in resize, crop, normalization, channel order, mask encoding, augmentation, or sample ordering invalidates prior CRP relevance statistics, PCX attribution banks, GMM prototypes, and reference-image caches.

### If architecture or explanation layer changes

1. Register/adapt the model in `LCRP/models/`.
2. Add or update its attributor, canonizer, composite, and visualization in `LCRP/utils/crp_configs.py`.
3. Inspect valid convolution names using `crp.helper.get_layer_names`.
4. Set exactly the same layer in CRP, PCX-bank generation, GMM fitting, plotting, and `src/explanator.py`.
5. Recalculate the CRP statistics, PCX attribution banks, GMM prototypes, and reference images.

Changing only the string in `Explanator` is insufficient: a new layer normally has a different channel count, making old attribution banks and GMMs incompatible.

### Explanation-data checklist

Before deployment, confirm:

- checkpoint and manifest checkpoint basename match;
- dataset length/order and length match;
- class order is correct;
- the selected layer exists and matches all CRP, PCX, GMM, and reference-image paths;
- every class bank is a finite two-dimensional array;
- each metadata row count equals its bank row count;
- the chosen prototype count matches each GMM cache;
- reference-image HDF5 files can be read;
- at least one end-to-end PCX plot succeeds for every supported class.

## Configure the runtime service

The runtime supports these environment variables (all relative defaults are
resolved from `PROJECT_ROOT`):

| Variable | Default |
|---|---|
| `PERSON_VEHICLE_CHECKPOINT` | `models/best_ckpt.pt` |
| `PERSON_VEHICLE_DATA_ROOT` | `data/BRK/person_vehicle_detection` |
| `PERSON_VEHICLE_CRP_DIR` | current validated CRP directory in `src/explanator.py` |
| `PERSON_VEHICLE_PCX_DIR` | current validated PCX directory in `src/explanator.py` |
| `PERSON_VEHICLE_REF_IMAGES_DIR` | `output/ref_imgs_yolov6_brk_validated` |
| `PERSON_VEHICLE_MIN_EXPLANATION_CONFIDENCE` | `0.30` |
| `FLOOD_CHECKPOINT` | `models/flood_model_brk2.pt` |
| `FLOOD_DATA_ROOT` | `data/BRK/flood_segmentation` |
| `FLOOD_CRP_DIR` | `examples/output/crp/CRP_BRK2` |
| `FLOOD_PCX_DIR` | `examples/output/pcx/PCX-BRK2` |
| `FLOOD_REF_IMAGES_DIR` | `examples/new-ref-img-BRK-FLOOD10-ALL` |
| `FLOOD_PCX_LAYER` | `layer5.0.conv1` |
| `FLOOD_DATA_SPLIT` | `train` |
| `FLOOD_MIN_COVERAGE` | `0.10` |
| `ENTITIES_TO_EXPLAIN` | `FloodSegmentation,PersonVehicleDetection` (left-to-right queue order) |
| `PORT` | `8080` |
| `BASE_PATH` | `/tfa02` |
| `REDIS_HOST` | `localhost` |
| `REDIS_PORT` | `6379` |
| `BROKER_URL` | TEMA Orion URL from the Dockerfile |

MinIO credentials, endpoint, and bucket behavior are currently constants in `src/minio_client.py`. Before deployment, migrate them to environment variables or a secret manager, configure them for the target environment, and rotate any credentials that have been exposed in source history.

Example local explanation-data configuration:

```bash
export PERSON_VEHICLE_CHECKPOINT="$PWD/models/best_ckpt.pt"
export PERSON_VEHICLE_DATA_ROOT="$PWD/data/BRK/person_vehicle_detection"
export PERSON_VEHICLE_CRP_DIR="$PWD/output/crp/yolo_person_car_full"
export PERSON_VEHICLE_PCX_DIR="$PWD/output/pcx/from_yolo_person_car_full"
export PERSON_VEHICLE_REF_IMAGES_DIR="$PWD/output/ref_imgs_yolo_person_car_full"
```

Select explanations and their order with a comma-separated list. Person and
car are produced together by `PersonVehicleDetection`:

```bash
# Flood first, then person/car (default)
ENTITIES_TO_EXPLAIN=FloodSegmentation,PersonVehicleDetection ./run_docker.sh

# Flood only
ENTITIES_TO_EXPLAIN=FloodSegmentation ./run_docker.sh

# Person/car first, then flood
ENTITIES_TO_EXPLAIN=PersonVehicleDetection,FloodSegmentation ./run_docker.sh
```

## Run and test locally

Run these in separate terminals:

```bash
redis-server
```

```bash
DEBUG=1 python app.py
```

```bash
python worker.py
```

Basic health checks:

```bash
curl -f http://localhost:8080/tfa02/ping
curl -s http://localhost:8080/tfa02/tasks
```

Run the focused unit tests from the repository root:

```bash
pytest -q tests/test_yolo_prototype_class_matching.py
pytest -q tests/test_yolov6_canonization.py
```

Run all tests:

```bash
pytest -q
```

The scripts `tests/test_post_data.py`, `tests/test_subscription.py`, `tests/get_entity.py`, and `tests/delete_entity.py` are integration utilities: they contact a running service and may contact external infrastructure. Review their endpoint/entity configuration before running them. The existing notification examples are documented in `tests/README.md`.

An end-to-end local request can be sent with:

```bash
python tests/test_post_data.py ImageMetadata
```

Then inspect:

```bash
curl -s http://localhost:8080/tfa02/tasks
docker logs explanation_tfa02 2>&1 | tail -n 200
```

For an explanation-data smoke test, generate one explanation from `examples/yolo_brk_pcx.ipynb` and verify that a readable PNG appears under the new `pcx_plots/` directory. This catches path, layer, bank, metadata, GMM, class, and reference-image incompatibilities before deploying the API.

## Docker deployment

Build the image:

```bash
docker build -t explanation_tfa02 .
```

For the TEMA amd64 target:

```bash
docker build --platform linux/amd64 -t explanation_tfa02 .
```

The image starts Redis, Flask, and one RQ worker through Supervisor. The supplied `run_docker.sh` requests the NVIDIA runtime and all GPUs:

```bash
./run_docker.sh
```

Equivalent command with explicit explanation-data mounts and configuration:

```bash
docker run --rm \
  --name explanation_tfa02 \
  --gpus all \
  -p 8080:8080 \
  -e PERSON_VEHICLE_CHECKPOINT=/explanation-data/models/best_ckpt.pt \
  -e PERSON_VEHICLE_DATA_ROOT=/explanation-data/data/person_vehicle_detection \
  -e PERSON_VEHICLE_CRP_DIR=/explanation-data/output/crp/yolo_person_car_full \
  -e PERSON_VEHICLE_PCX_DIR=/explanation-data/output/pcx/from_yolo_person_car_full \
  -e PERSON_VEHICLE_REF_IMAGES_DIR=/explanation-data/output/ref_imgs_yolo_person_car_full \
  -e FLOOD_CHECKPOINT=/explanation-data/models/flood_model_brk2.pt \
  -e FLOOD_DATA_ROOT=/explanation-data/data/flood_segmentation \
  -e FLOOD_CRP_DIR=/explanation-data/output/crp/CRP_BRK2 \
  -e FLOOD_PCX_DIR=/explanation-data/output/pcx/PCX-BRK2 \
  -e FLOOD_REF_IMAGES_DIR=/explanation-data/output/ref-images/flood \
  -e ENTITIES_TO_EXPLAIN=FloodSegmentation,PersonVehicleDetection \
  -v /absolute/host/explanation-data:/explanation-data:rw \
  explanation_tfa02
```

After startup:

```bash
curl -f http://localhost:8080/tfa02/ping
docker logs -f explanation_tfa02
```

The code can select CPU when CUDA is unavailable, and the PIDNet path has an out-of-memory CPU fallback. Full CRP/PCX generation is nevertheless computationally expensive; a CUDA-capable deployment is recommended. Verify that the container's PyTorch build can see the GPU:

```bash
docker exec explanation_tfa02 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

To publish, retain the GHCR steps in the original README above, use an immutable version tag, push it, and provide the deployment operator with the exact explanation-data version and required environment variables. Do not promote an image merely because `/ping` succeeds; also send a representative entity and wait for its queued task to complete.

## Troubleshooting

- **No samples accepted for CRP:** inspect class order, prediction quality, annotation geometry, and IoU threshold. Use `--scan-limit` for quick diagnostics, but do not lower IoU without checking examples.
- **Empty/non-differentiable detection:** CRP must run on the validated `DetectionSubset`; images with empty post-NMS output are intentionally excluded.
- **Manifest mismatch:** checkpoint, dataset size/order, or CRP/PCX results differ. Recalculate the matching CRP statistics, PCX banks, GMM prototypes, and reference images.
- **Missing layer:** print available convolution layers and use the exact canonized model name. Layer strings are architecture-specific.
- **GMM feature mismatch:** the PCX bank came from another layer/model. Do not reshape it; rebuild it.
- **Cached prototype-count mismatch:** select a new PCX output directory or remove only the clearly identified stale cache, then refit.
- **CUDA out of memory:** use one CRP layer, keep batch size at one, reduce displayed references/concepts for inference, or use the implemented CPU fallback where available.
- **New image produces no explanation:** the detector may have no valid boxes or all confidence values may be below `PERSON_VEHICLE_MIN_EXPLANATION_CONFIDENCE`.
- **Notebook imports fail:** launch Jupyter from the repository root and replace old absolute paths in the first configuration cell.
