# TEMA TFA-02 — Concept-based Explanations

This repository contains the currently deployed TFA-02 explanation service for the TEMA project. The service applies concept-based explainable AI methods to predictions produced by supported computer-vision components and publishes explanation results through the TEMA infrastructure.

The `main` branch documents and represents the shared deployed service. Trial-specific data preparation, model checkpoints, explanation parameters, notebooks, and operational instructions belong in their corresponding trial branches.

## Branch policy

- `main` contains the shared application, deployment configuration, service interface, and general documentation.
- Each trial branch contains the dataset paths, model weights, preprocessing, class mappings, CRP/PCX calculations, selected layers, prototypes, and trial-specific notebooks needed for that trial.
- Start new trial work from the deployed baseline, but keep trial-only changes on a dedicated branch.
- Merge a trial change into `main` only when it is intended to become part of the common deployed service.

To work on a trial:

```bash
git fetch origin
git branch -r
git switch <trial-branch>
git pull --ff-only
```

Do not copy explanation results between trials unless the checkpoint, dataset ordering, preprocessing, model architecture, class mapping, selected layer, and all explanation settings are identical.

## Explanation methods

The service builds on four related methods:

1. **Layer-wise Relevance Propagation (LRP)** explains a selected output by redistributing its prediction score backward through the network. At input level, the resulting relevance values show how individual input features contributed to the prediction.
2. **Concept Relevance Propagation (CRP)** extends LRP by conditioning the backward relevance flow on hidden-layer units. In convolutional models, channels can be interpreted as latent concepts. CRP identifies relevant concepts and produces concept-conditional heatmaps showing where they occur. Relevance Maximization provides representative dataset examples for which a concept was useful to the model.
3. **L-CRP** adapts CRP to localization models. Object-detection explanations target the class score of a selected bounding box; segmentation explanations target a selected class region or region of interest.
4. **Prototypical Concept-based Explanations (PCX)** represents decisions with concept-relevance vectors and groups recurring class-wise decision strategies into prototypes. Comparing a prediction with these prototypes helps identify the strategy used by the model and behavior that deviates from typical concept use.

The general offline-to-online flow is:

```text
dataset + labels + checkpoint + preprocessing
                        |
                        v
            validate model predictions
                        |
                        v
            calculate CRP statistics
                        |
                        v
       calculate PCX attribution banks
                        |
                        v
       fit prototypes and collect concept
              reference examples
                        |
                        v
       deploy the matching explanation data
                        |
                        v
        explain new model predictions
```

CRP statistics, PCX attribution banks, prototype models, metadata, and reference-image caches are coupled to the model and preprocessing used to create them. Recalculate the complete set when any of those inputs changes.

## Service architecture

The deployed component consists of:

```text
TEMA notification
       |
       v
Flask API (`app.py`)
       |
       v
Redis/RQ queue
       |
       v
Worker (`worker.py` and `tasks.py`)
       |
       +--> download input from MinIO
       +--> run `src.explanator.Explanator`
       +--> upload explanation results
       +--> update task state and external entity
```

The Docker image runs Flask, Redis, and an RQ worker under Supervisor.

### HTTP endpoints

The default base path is `/tfa02`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/tfa02` | `GET` | Component landing page |
| `/tfa02/ping` | `GET` | Health check |
| `/tfa02/post_data` | `POST` | Receive an entity or notification and enqueue supported work |
| `/tfa02/task_status/<task_id>` | `GET` | Read one queued task's status |
| `/tfa02/tasks` | `GET` | List tracked tasks and queue length |
| `/tfa02/clean_tasks` | `GET` | Clear stored task records |
| `/tfa02/requeue_tasks` | `GET` | Requeue eligible incomplete tasks |

The task-management endpoints change queue state and should be protected appropriately in a production environment.

## Repository structure

| Path | Purpose |
|---|---|
| [`app.py`](app.py) | Flask API, request validation, and queue submission |
| [`tasks.py`](tasks.py) | Queued image processing, storage operations, result publication, and status updates |
| [`worker.py`](worker.py) | RQ worker entry point |
| [`common_app_funcs.py`](common_app_funcs.py) | Redis, queue, task-state, and entity-update helpers |
| [`src/explanator.py`](src/explanator.py) | Runtime model loading and explanation orchestration |
| [`src/entities.py`](src/entities.py) | Explanation entity templates and builders |
| [`src/glocal_analysis.py`](src/glocal_analysis.py) | Dataset-wide CRP calculation |
| [`src/plot_crp_explanations.py`](src/plot_crp_explanations.py) | CRP explanation rendering |
| [`src/plot_pcx_explanations_YOLO.py`](src/plot_pcx_explanations_YOLO.py) | Object-detection PCX rendering |
| [`src/plotpcx_gpu.py`](src/plotpcx_gpu.py) | Device-aware segmentation PCX rendering |
| [`src/pcx_helper.py`](src/pcx_helper.py) | Reference-image, crop, prototype, and visualization helpers |
| [`src/device_utils.py`](src/device_utils.py) | PyTorch device resolution |
| [`src/letterbox_utils.py`](src/letterbox_utils.py) | Object-detector preprocessing and box rescaling |
| [`src/kpi_logging.py`](src/kpi_logging.py) | Timing and KPI records |
| [`src/memory_logging.py`](src/memory_logging.py) | CUDA memory diagnostics |
| [`src/minio_client.py`](src/minio_client.py) | MinIO access |
| [`src/datasets/`](src/datasets) | Dataset adapters and preprocessing |
| [`LCRP/models/`](LCRP/models) | Supported model wrappers |
| [`LCRP/utils/crp_configs.py`](LCRP/utils/crp_configs.py) | Attributors, canonizers, composites, and feature visualizations |
| [`examples/`](examples) | Research and trial preparation; follow the selected trial branch |
| [`tests/`](tests) | Unit tests and service integration utilities |
| [`Dockerfile`](Dockerfile) | Production container definition |
| [`supervisord.conf`](supervisord.conf) | Flask, Redis, and worker process configuration |
| [`run_docker.sh`](run_docker.sh) | GPU-enabled local container command |

Several research and legacy implementations remain for reproducibility. The deployed behavior is determined by the imports in `src/explanator.py`; changing an unused experimental module does not change the service.

## Configuration

The container defines these general defaults:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | Flask service port |
| `BASE_PATH` | `/tfa02` | URL prefix |
| `BROKER_URL` | TEMA broker URL in the container | Context broker used for entity updates |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `PROCESSING_UNIT` | `gpu` | Intended processing mode |
| `DEBUG` | `False` | Application debug setting |

Model paths, dataset roots, CRP directories, PCX directories, reference-image directories, confidence thresholds, and other explanation settings depend on the deployed trial. Configure them according to that trial branch and `src/explanator.py`.

MinIO connection values are currently defined in `src/minio_client.py`. Production credentials should be provided through environment variables or a secret manager, not committed to source control. Rotate any credential that has been exposed in repository history.

## Build the Docker image

To build for the TEMA cloud:

```bash
docker build --platform linux/amd64 -t explanation_tfa02 .
```

For development on a Mac, build like this:

```bash
docker build -t explanation_tfa02 .
```

## Push the image to the registry

Obtain a Personal Access Token (PAT) by following [these instructions](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

It is likely that you already have a PAT stored in your Git credential helper. To see it, run:

```bash
echo "url=https://github.com" | git credential fill
```

Set the PAT environment variable to your PAT value:

```bash
export PAT=<your_git_personal_access_token>
```

Test the GHCR login:

```bash
echo $PAT | docker login ghcr.io -u <your_git_username> --password-stdin
```

Push the TFA-02 container image to the Container Registry:

```bash
docker tag explanation_tfa02 ghcr.io/he-tema/explanation_tfa02:1.0
docker push ghcr.io/he-tema/explanation_tfa02:1.0
```

Write to Nicola Colosi on TEMA Slack to deploy the pushed container on the TEMA cluster.

## Run the Docker container

Edit the file if you want to change some environment variables:

```bash
./run_docker.sh
```

## Development

### Install dependencies

```bash
conda create -n tema python=3.8
conda activate tema
pip install -r requirements.txt
```

During development, you can run the application components separately in different terminal tabs for easier debugging and log monitoring.

Start the Redis server:

```bash
redis-server
```

Start the Flask application:

```bash
DEBUG=1 python app.py
```

`DEBUG=1` will make the application reload on code changes. Remove it if you do not want automatic reloading.

Start the worker process:

```bash
python worker.py
```

For the worker to work properly on a Mac, run this before starting it:

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

The application will be available at `http://localhost:8080/tfa02` by default.

For production deployment, use the Docker container as described in the [Run the Docker container](#run-the-docker-container) section above.

## Testing

You can test the application using the provided test script in the `tests/` folder:

```bash
cd tests
```

There are two ways to run the tests.

Local testing sends notifications to local Redis:

```bash
python test_post_data.py ImageMetadata
```

Cloud testing sends notifications to the TEMA cloud:

```bash
python test_post_data.py ImageMetadata --cloud
```

The test script sends sample image metadata to the application. You should see the processing results in the logs of both the Flask application and the worker process.

## Logs and operations

Follow container logs:

```bash
docker logs -f explanation_tfa02
```

Show recent logs:

```bash
docker logs explanation_tfa02 2>&1 | tail -n 200
```

Inspect tasks:

```bash
curl -s http://localhost:8080/tfa02/tasks
curl -s http://localhost:8080/tfa02/task_status/<task-id>
```

Common problems:

- **Task remains queued:** verify Redis connectivity and that the RQ worker is listening to the `image_processing` queue.
- **Input cannot be downloaded:** verify the MinIO endpoint, credentials, bucket, and object name.
- **Model or explanation data cannot be loaded:** verify all configured paths and confirm that they belong to the same trial, checkpoint, preprocessing, class mapping, and selected layer.
- **No explanation is produced:** inspect prediction availability and any configured confidence or target-selection rules.
- **CUDA out of memory:** inspect memory logs, reduce trial-configurable explanation complexity where scientifically acceptable, or use an implemented fallback.
- **Entity update fails:** verify broker connectivity, entity schema, required identifiers, and authorization.

## Preparing a new trial

Trial preparation belongs on a dedicated branch:

```bash
git switch main
git pull --ff-only
git switch -c <trial-name>
```

On that branch:

1. add or configure the trial dataset adapter;
2. configure the checkpoint and model output classes;
3. reproduce the exact inference preprocessing offline;
4. validate predictions and target selection;
5. calculate CRP relevance statistics;
6. calculate class-specific PCX attribution banks and metadata;
7. select and validate the prototype count;
8. generate concept reference images;
9. test at least one complete explanation for every supported output class;
10. configure the runtime paths and settings;
11. run unit, local integration, and authorized cloud tests;
12. document all trial-specific choices in that branch's README.

Recalculate the complete explanation data when the checkpoint, architecture, class order, dataset content or ordering, preprocessing, selected layer, or concept representation changes.

## Scientific references

### LRP — Layer-wise Relevance Propagation

- Bach et al., [*On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation*](https://doi.org/10.1371/journal.pone.0130140)
- Montavon et al., [*Layer-Wise Relevance Propagation: An Overview*](https://doi.org/10.1007/978-3-030-28954-6_10)
- [Zennit toolbox](https://github.com/chr5tphr/zennit)

### CRP — Concept Relevance Propagation

- Achtibat et al., [*From Attribution Maps to Human-Understandable Explanations through Concept Relevance Propagation*](https://doi.org/10.1038/s42256-023-00711-8)
- [Zennit-CRP toolbox](https://github.com/rachtibat/zennit-crp)

### L-CRP — CRP for Localization Models

- Dreyer et al., [*Revealing Hidden Context Bias in Segmentation and Object Detection through Concept-specific Explanations*](https://arxiv.org/pdf/2211.11426)
- [L-CRP code](https://github.com/maxdreyer/L-CRP/tree/main)

### PCX — Prototypical Concept-based Explanations

- Dreyer et al., [*Understanding the (Extra-)Ordinary: Validating Deep Model Decisions with Prototypical Concept-based Explanations*](https://arxiv.org/pdf/2311.16681)
- [PCX code](https://github.com/maxdreyer/pcx/tree/main)
