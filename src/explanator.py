import gc
import json
import os
import torch
import torchvision.transforms as transforms
import numpy as np
# Set non-interactive backend for matplotlib to avoid GUI issues in Flask
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import logging
from contextlib import contextmanager
from functools import partial
from statistics import mean

from LCRP.models import get_model
from src.plot_crp_explanations import plot_one_image_explanation, fig_to_array
from src.plot_pcx_explanations_YOLO import plot_pcx_explanations
from src.plot_pcx_explanations_YOLO import plot_one_image_pcx_explanation
from src.plot_pcx_explanations_YOLO import ExplanationUnavailableError
from src.plotpcx_gpu import plot_pcx_explanations_pidnet
from src.datasets.person_car_dataset import PersonCarDataset
from src.datasets.detection_subset import DetectionSubset
from src.datasets.flood_dataset import FloodDataset
from src.entities import (
    get_flood_segmentation_explanation_entity,
    get_person_vehicle_detection_explanation_entity,
)
from src.minio_client import FHHI_MINIO_BUCKET
from src.memory_logging import log_cuda_memory
from src.letterbox_utils import YOLOv6TrainPreprocess, letterbox_transform, rescale_boxes
from src.yolo_class_mapping import resolve_display_class_names
from src.kpi_logging import (
    append_avg_window_record,
    append_kpi_record,
    build_kpi_record,
    build_log_path,
    timed_section,
)



def _empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()

class Explanator:
    """Class that stores all loaded models together with all relevant data for generating CRP explanations.

    This is the main class used in the TFA-02 component.
    """

    def __init__(self, project_root: str, logger: logging.Logger):
        self.logger = logger
        # General setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        # Log initial memory state
        log_cuda_memory(self.logger, "INIT")

        self.project_root = project_root
        self.person_vehicle_checkpoint = os.environ.get(
            "PERSON_VEHICLE_CHECKPOINT",
            os.path.join(self.project_root, "models", "best_ckpt.pt"),
        )
        self.person_vehicle_data_root = os.environ.get(
            "PERSON_VEHICLE_DATA_ROOT",
            os.path.join(
                self.project_root,
                "data",
                "BRK",
                "person_vehicle_detection",
            ),
        )
        self.person_vehicle_crp_dir = os.environ.get(
            "PERSON_VEHICLE_CRP_DIR",
            os.path.join(
                self.project_root,
                "output",
                "crp",
                "yolov6_validated_full_one_layer_20260727_145542",
            ),
        )
        self.person_vehicle_pcx_dir = os.environ.get(
            "PERSON_VEHICLE_PCX_DIR",
            os.path.join(
                self.project_root,
                "output",
                "pcx",
                "from_yolov6_validated_full_one_layer_20260727_145542",
            ),
        )
        self.person_vehicle_ref_images_dir = os.environ.get(
            "PERSON_VEHICLE_REF_IMAGES_DIR",
            os.path.join(
                self.project_root,
                "output",
                "ref_imgs_yolov6_brk_validated",
            ),
        )
        self.kpi_mirror_roots = []
        mirror_root = "/home/heydari/Jawher/FHHI-XAI"
        if os.path.abspath(self.project_root) != os.path.abspath(mirror_root):
            self.kpi_mirror_roots.append(mirror_root)

        # Lazy loading approach - don't load models until needed
        self._person_vehicle_model = None
        self._person_car_dataset = None
        self._person_car_crp_dataset = None
        self._person_car_dataset_orig = None
        self._flood_model = None
        self._flood_dataset = None

        # Create a mapping from entity types to handler methods
        self.entity_handlers = {
            "BurntSegmentation": self.explain_burnt_segmentation,
            "FireSegmentation": self.explain_fire_segmentation,
            "FloodSegmentation": self.explain_flood_segmentation,
            "PersonVehicleDetection": self.explain_person_vehicle_detection,
            "SmokeSegmentation": self.explain_smoke_segmentation,
            "EOBurntArea": self.explain_eo_burnt_area,
            "EOFloodExtent": self.explain_eo_flood_extent,
            "ImageMetadata": None,
        }

        self.VALID_ENTITY_TYPES = list(self.entity_handlers.keys())
        self.DLR_ENTITY_TYPES = {"EOBurntArea", "EOFloodExtent"}

        self.running_avg_forward_time = 0
        self.forward_count = 0
        self.running_avg_backward_time = 0
        self.backward_count = 0
        self._last_forward_time_ms = 0.0
        self.latest_kpi_log_paths = []

    def _log_path_variants(self, filename: str):
        roots = [self.project_root] + list(self.kpi_mirror_roots)
        seen = set()
        paths = []
        for root in roots:
            path = build_log_path(root, filename)
            if path not in seen:
                seen.add(path)
                paths.append(path)
        return paths

    def _append_kpi_record_all(self, filename: str, record: dict):
        paths = self._log_path_variants(filename)
        for path in paths:
            append_kpi_record(path, record)
        return paths

    def _append_avg_window_record_all(self, source_filename: str, avg_filename: str, **kwargs):
        source_paths = self._log_path_variants(source_filename)
        avg_paths = self._log_path_variants(avg_filename)
        for source_path, avg_path in zip(source_paths, avg_paths):
            append_avg_window_record(
                source_log_path=source_path,
                avg_log_path=avg_path,
                **kwargs,
            )
        return source_paths, avg_paths

    @property
    def prediction_times(self):
        """Returns the average forward and backward pass times."""
        return {
            "forward": f"{self.running_avg_forward_time:.3f} ms",
            "backward": f"{self.running_avg_backward_time:.3f} ms",
        }

    @contextmanager
    def record_forward_time(self):
        """Context manager to record the time taken for a forward pass."""
        if self.device == "cuda" and torch.cuda.is_available():
            self.logger.debug("Using CUDA for timing")
            try:
                # Try using CUDA events for timing on GPU
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                yield
                end_time.record()
                # Wait for the events to be recorded
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
            except (TypeError, RuntimeError):
                # Fall back to time.time() if CUDA events fail
                import time
                start_time = time.time()
                yield
                elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        else:
            self.logger.debug("Using CPU for timing")
            print()
            # Use time.time() for timing on CPU
            import time
            start_time = time.time()
            yield
            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # The formula for the running average is:
        # new_average = old_average + (new_value - old_average) / new_count
        self.forward_count += 1
        self.running_avg_forward_time += (elapsed_time - self.running_avg_forward_time) / self.forward_count
        self._last_forward_time_ms = elapsed_time
        self.logger.debug(f"Forward pass time: {elapsed_time:.2f} ms")
        self.logger.debug(f"Running average forward pass time: {self.running_avg_forward_time:.2f} ms")

    def explain(self, entity_type: str, original_image_bucket: str, original_image_filename: str, image: np.ndarray, bm_id, uav_id, flight_number, alert_ref):
        """Generate explanation for the given entity type and image."""
        log_cuda_memory(self.logger, f"BEFORE EXPLAIN {entity_type}")
        self.latest_kpi_log_paths = []

        if entity_type not in self.VALID_ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {entity_type}. Must be one of {self.VALID_ENTITY_TYPES}")

        # Get the appropriate handler method for this entity type
        handler = self.entity_handlers.get(entity_type)

        # Call the handler method with the image
        result = handler(original_image_bucket, original_image_filename, image, bm_id=bm_id, uav_id=uav_id, flight_number=flight_number, alert_ref=alert_ref)

        log_cuda_memory(self.logger, f"AFTER EXPLAIN {entity_type}")
        # Clear unnecessary tensors from cache
        _empty_cuda_cache()

        return result

    def explain_eo_burnt_area(self, original_image_bucket: str, original_image_filename: str, image: np.ndarray):
        raise NotImplementedError("EO Burnt Area explanation is not implemented yet.")

    def explain_eo_flood_extent(self, original_image_bucket: str, original_image_filename: str, image: np.ndarray):
        raise NotImplementedError("EO Flood Extent explanation is not implemented yet.")

    def explain_burnt_segmentation(self, original_image_bucket: str, original_image_filename: str, image: np.ndarray):
        raise NotImplementedError("Burnt segmentation explanation is not implemented yet.")

    def explain_fire_segmentation(self, original_image_bucket: str, original_image_filename: str, image: np.ndarray):
        raise NotImplementedError("Fire segmentation explanation is not implemented yet.")

    @property
    def flood_model(self):
        if self._flood_model is None:
            log_cuda_memory(self.logger, "BEFORE LOADING FLOOD MODEL")
            model_name = "pidnet"
            # flood_model_path = os.path.join(self.project_root, "models", "flood_s_best_pidnet_modified.pt")
            self._flood_model = get_model(model_name=model_name)
            self._flood_model.eval()
            log_cuda_memory(self.logger, "AFTER LOADING FLOOD MODEL")
        return self._flood_model

    @property
    def flood_dataset(self):
        if self._flood_dataset is None:
            flood_data_path = os.path.join(self.project_root, "data", "General_Flood_v3")

            target_dtype = self.dtype
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.to(dtype=target_dtype) if isinstance(x, torch.Tensor) else x),
            ])

            self._flood_dataset = FloodDataset(root_dir=flood_data_path, split="train", transform=transform)
        return self._flood_dataset

    def explain_flood_segmentation(self, original_image_bucket: str, original_image_filename: str, image: np.ndarray, bm_id, uav_id, flight_number, alert_ref):
        """Generate flood segmentation explanation using PCX."""
        log_cuda_memory(self.logger, "FLOOD_SEG START")
        original_entity_type = "FloodSegmentation"

        # Parameters
        class_id = 1  # Flood class ID
        n_concepts = 3
        n_refimgs = 12
        model_name = "pidnet"
        num_prototypes = 2
        output_dir_pcx = "output/pcx/pidnet_flood/"
        output_dir_crp = "output/crp/pidnet_flood/"
        ref_imgs_path = "output/ref_imgs_pidnet/"
        # layer_names = get_layer_names(self.flood_model, [torch.nn.Conv2d])
        layer_name = 'layer5.0.conv1'
        print(layer_name)
        # Apply transform to the input test image
        log_cuda_memory(self.logger, "BEFORE IMAGE TRANSFORM")
        image_tensor = self.flood_dataset.transform(image)
        image_tensor = image_tensor.to(self.device, non_blocking=True)

        log_cuda_memory(self.logger, "AFTER IMAGE TRANSFORM")

        print("Shape after batch dimension:", image_tensor.shape)
        self.flood_model.eval()

        with timed_section(self.device) as prediction_timer:
            with torch.no_grad():
                _ = self.flood_model(image_tensor.unsqueeze(0))
        prediction_time_s = prediction_timer.elapsed_s

        log_cuda_memory(self.logger, "BEFORE EXPLANATION GENERATION")
        used_n_refimgs = n_refimgs
        used_n_concepts = n_concepts
        try:
            explanation_fig = plot_pcx_explanations_pidnet(
                model_name,
                self.flood_model,
                self.flood_dataset,
                image_tensor=image_tensor,
                layer_name=layer_name,
                n_concepts=n_concepts,
                n_refimgs=n_refimgs,
                num_prototypes=num_prototypes,
                ref_imgs_path=ref_imgs_path,
                output_dir_crp=output_dir_crp,
                output_dir_pcx=output_dir_pcx,
                precision="autocast_fp16" if (self.device == "cuda" and torch.cuda.is_available()) else "fp32",
            )
        except Exception as exc:
            if not (_is_cuda_oom(exc) and self.device == "cuda" and torch.cuda.is_available()):
                raise
            self.logger.warning("Flood explanation hit CUDA OOM; retrying on CPU: %s", exc)
            gc.collect()
            _empty_cuda_cache()
            cpu_model = self.flood_model.to("cpu")
            cpu_model.eval()
            explanation_fig = plot_pcx_explanations_pidnet(
                model_name,
                cpu_model,
                self.flood_dataset,
                image_tensor=image_tensor.detach().cpu(),
                layer_name=layer_name,
                n_concepts=n_concepts,
                n_refimgs=n_refimgs,
                num_prototypes=num_prototypes,
                ref_imgs_path=ref_imgs_path,
                output_dir_crp=output_dir_crp,
                output_dir_pcx=output_dir_pcx,
                device=torch.device("cpu"),
                precision="fp32",
            )
            self._flood_model = cpu_model
            used_n_refimgs = getattr(explanation_fig, "_n_refimgs_used", n_refimgs)
            used_n_concepts = getattr(explanation_fig, "_n_concepts_used", n_concepts)
        finally:
            # Release the input tensor as soon as the attribution run finishes
            del image_tensor
            _empty_cuda_cache()

        used_n_refimgs = getattr(explanation_fig, "_n_refimgs_used", n_refimgs)
        used_n_concepts = getattr(explanation_fig, "_n_concepts_used", n_concepts)
        figure_kpis = getattr(explanation_fig, "_kpi_metrics", {})
        backward_time_s = float(figure_kpis.get("backward_time_s", 0.0))
        full_attribution_time_s = float(figure_kpis.get("full_attribution_time_s", 0.0))

        pidnet_record = build_kpi_record(
            model="pidnet",
            entity_type=original_entity_type,
            scope="image",
            aggregation="raw",
            image=original_image_filename,
            prediction_time_s=prediction_time_s,
            global_lcrp_time_s=backward_time_s,
            global_total_time_s=full_attribution_time_s,
            layer=layer_name,
            n_concepts=used_n_concepts,
            n_refimgs=used_n_refimgs,
        )
        pidnet_image_logs = self._append_kpi_record_all("pidnet_image_kpis.txt", pidnet_record)
        _, pidnet_avg5_logs = self._append_avg_window_record_all(
            source_filename="pidnet_image_kpis.txt",
            avg_filename="pidnet_image_kpis_avg5.txt",
            model="pidnet",
            entity_type=original_entity_type,
            scope="image",
            layer=layer_name,
        )
        self.latest_kpi_log_paths = pidnet_image_logs + pidnet_avg5_logs

        # fig is returned implicitly as part of this function; adapt if needed
        log_cuda_memory(self.logger, "AFTER EXPLANATION GENERATION")

        explanation_img = fig_to_array(explanation_fig)
        plt.close(explanation_fig)
        gc.collect()

        # Prepare explanation entity
        explanation_image_filename = f"tfa02/{original_entity_type}/{original_image_filename}"

        explanation_entity = get_flood_segmentation_explanation_entity(
            original_image_bucket=original_image_bucket,
            original_image_filename=original_image_filename,
            explanation_image_bucket=FHHI_MINIO_BUCKET,
            explanation_image_filename=explanation_image_filename,
            class_id=class_id,
            n_concepts=used_n_concepts,
            n_refimgs=used_n_refimgs,
            layer=layer_name,
            mode="relevance",
            bm_id=bm_id,
            uav_id=uav_id,
            flight_number=flight_number,
            alert_ref=alert_ref
        )

        log_cuda_memory(self.logger, "FLOOD_SEG END")
        _empty_cuda_cache()

        return explanation_entity, [explanation_img], [explanation_image_filename]

    @property
    def person_vehicle_model(self):
        if self._person_vehicle_model is None:
            log_cuda_memory(self.logger, "BEFORE LOADING PERSON VEHICLE MODEL")
            self._person_vehicle_model = self.load_person_vehicle_model()
            log_cuda_memory(self.logger, "AFTER LOADING PERSON VEHICLE MODEL")
        return self._person_vehicle_model

    @property
    def person_car_dataset(self):
        if self._person_car_dataset is None:
            self._person_car_dataset = self.load_person_car_data()
        return self._person_car_dataset

    @property
    def person_car_dataset_orig(self):
        """Dataset without transform for accessing original high-res images."""
        if not hasattr(self, '_person_car_dataset_orig') or self._person_car_dataset_orig is None:
            self._person_car_dataset_orig = PersonCarDataset(
                root_dir=self.person_vehicle_data_root,
                split="train",
                transform=transforms.ToTensor(),
            )
        return self._person_car_dataset_orig

    @property
    def person_car_crp_dataset(self):
        """Exact manifest-backed dataset used to compute the PCX artifacts."""
        if self._person_car_crp_dataset is None:
            manifest_path = os.path.join(
                self.person_vehicle_crp_dir,
                "crp_filter_manifest.json",
            )
            if not os.path.isfile(manifest_path):
                raise FileNotFoundError(
                    f"Missing CRP selection manifest: {manifest_path}"
                )
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)

            if int(manifest["dataset_length"]) != len(self.person_car_dataset):
                raise ValueError(
                    f"CRP manifest dataset length {manifest['dataset_length']} "
                    f"does not match runtime dataset length {len(self.person_car_dataset)}."
                )
            manifest_checkpoint = os.path.basename(manifest.get("checkpoint", ""))
            runtime_checkpoint = os.path.basename(self.person_vehicle_checkpoint)
            if manifest_checkpoint != runtime_checkpoint:
                raise ValueError(
                    f"CRP checkpoint {manifest_checkpoint!r} does not match "
                    f"runtime checkpoint {runtime_checkpoint!r}."
                )

            indices = [int(index) for index in manifest["matched_indices"]]
            predicted_classes = {
                int(index): tuple(int(class_id) for class_id in class_ids)
                for index, class_ids in manifest["matched_classes"].items()
            }
            self._person_car_crp_dataset = DetectionSubset(
                dataset=self.person_car_dataset,
                indices=indices,
                predicted_classes=predicted_classes,
            )
            if len(self._person_car_crp_dataset) != int(
                manifest["crp_dataset_length"]
            ):
                raise ValueError("Runtime CRP subset length differs from the manifest.")
        return self._person_car_crp_dataset

    def load_person_vehicle_model(self):
        # Load the person/vehicle detection model
        model_name = "yolov6s6"
        model = get_model(model_name=model_name, classes=2, ckpt_path=self.person_vehicle_checkpoint, device=self.device,
                          dtype=self.dtype)
        model.eval()
        return model

    def load_person_car_data(self):
        transform = YOLOv6TrainPreprocess(target_size=640, stride=32, half=False)
        dataset = PersonCarDataset(
            root_dir=self.person_vehicle_data_root,
            split="train",
            transform=transform,
        )
        return dataset

    def _find_person_car_sample_id(self, original_image_filename: str):
        basename = os.path.basename(original_image_filename)
        target_stem = os.path.splitext(basename)[0].lower()
        for crp_index, original_index in enumerate(
            self.person_car_crp_dataset.indices
        ):
            image_file = self.person_car_dataset.image_files[original_index]
            if os.path.splitext(image_file)[0].lower() == target_stem:
                return crp_index
        return None

    def explain_person_vehicle_detection(self, original_image_bucket: str, original_image_filename: str,
                                         image: np.ndarray, bm_id, uav_id, flight_number, alert_ref):
        """Generate person/vehicle detection explanation."""
        original_entity_type = "PersonVehicleDetection"
        original_filename_no_ext = os.path.splitext(original_image_filename)[0]
        min_explanation_confidence = float(
            os.environ.get(
                "PERSON_VEHICLE_MIN_EXPLANATION_CONFIDENCE",
                "0.30",
            )
        )
        if not 0.0 <= min_explanation_confidence <= 1.0:
            raise ValueError(
                "PERSON_VEHICLE_MIN_EXPLANATION_CONFIDENCE must be between 0 and 1."
            )

        log_cuda_memory(self.logger, "PERSON_VEHICLE START")

        model_name = "yolov6s6"
        n_concepts = 3
        n_refimgs = 12
        layer = 'module.backbone.ERBlock_3.0.rbr_dense.conv'
        # More car components expose distinct appearance/context modes such as
        # white, dark, isolated, small and occluded vehicles.
        prototype_dict = {0: 4, 1: 5}

        mode = "relevance"
        crp_output_dir = self.person_vehicle_crp_dir
        pcx_output_dir = self.person_vehicle_pcx_dir
        ref_imgs_path = self.person_vehicle_ref_images_dir
        display_class_names = resolve_display_class_names(
            model_name,
            self.person_car_dataset,
        )
        if tuple(display_class_names) != ("person", "car"):
            raise ValueError(
                "The YOLO explanator requires detector classes "
                "0=person and 1=car to match the validated CRP/PCX artifacts."
            )
        sample_id = self._find_person_car_sample_id(original_image_filename)

        # Get original image shape (H, W)
        original_shape = image.shape[:2]

        log_cuda_memory(self.logger, "BEFORE IMAGE TRANSFORM")

        image_tensor = letterbox_transform(
            image,
            target_size=640,
            stride=32,
            half=False,
            auto=False,
            scaleup=True,
        )

        # Get ACTUAL letterbox shape from tensor (C, H, W) -> (H, W)
        letterbox_shape = (image_tensor.shape[1], image_tensor.shape[2])

        test_img = image_tensor.unsqueeze(0).to(self.device)

        with self.record_forward_time():
            scores, boxes = self.person_vehicle_model.predict_with_boxes(test_img)
        prediction_time_s = self._last_forward_time_ms / 1000.0
        num_boxes = boxes.shape[1]
        self.logger.debug(f"Number of boxes: {num_boxes}")

        if scores.numel() == 0 or boxes.numel() == 0 or num_boxes == 0:
            self.logger.info("No detections found for %s; skipping PCX and entity generation.", original_image_filename)
            log_cuda_memory(self.logger, "PERSON_VEHICLE NO DETECTIONS")
            _empty_cuda_cache()
            return None, [], []

        # Rescale boxes to original image coordinates
        boxes_np = boxes[0].cpu().detach().numpy()  # Shape: [N, 4]
        boxes_rescaled = rescale_boxes(boxes_np, letterbox_shape, original_shape)
        class_ids_all = scores[0].argmax(dim=1)
        confidences_all = scores[0].max(dim=1).values
        box_areas = (boxes[0][:, 2] - boxes[0][:, 0]) * (boxes[0][:, 3] - boxes[0][:, 1])
        # Run the detector forward pass once, then only spend backward/PCX work
        # on detections that are both real and confident enough to explain.
        valid_detection_mask = (box_areas > 0) & (confidences_all >= min_explanation_confidence)
        valid_detection_indices = valid_detection_mask.nonzero(as_tuple=True)[0]

        if valid_detection_indices.numel() == 0:
            self.logger.info(
                "No detections with confidence >= %.2f remained for %s; skipping PCX, plots, and entity generation.",
                min_explanation_confidence,
                original_image_filename,
            )
            log_cuda_memory(self.logger, "PERSON_VEHICLE NO VALID DETECTIONS")
            _empty_cuda_cache()
            return None, [], []

        boxes_list = boxes_rescaled[valid_detection_indices.cpu().numpy()].astype(int).tolist()
        class_ids = class_ids_all[valid_detection_indices]
        confidences = confidences_all[valid_detection_indices]
        num_boxes = int(valid_detection_indices.numel())

        explanation_images = []
        explanation_image_filenames = []
        explanation_boxes = []

        box_backward_times = []
        box_full_times = []
        for prediction_num, original_detection_index in enumerate(
            valid_detection_indices.tolist()
        ):
            class_id = class_ids[prediction_num].item()
            predicted_class_name = display_class_names[class_id]
            confidence = confidences[prediction_num].item()
            # Preserve the class-local rank in the detector's unfiltered
            # output. Filtering a lower-confidence object must never shift the
            # attribution onto another box of the same class.
            class_local_prediction_num = int(
                (class_ids_all[:original_detection_index + 1] == class_id)
                .sum()
                .item()
                - 1
            )

            self.logger.debug(f"Generating explanation for box {prediction_num} of {num_boxes}")
            self.logger.debug(
                "Prediction/prototype class locked to %s (%s)",
                class_id,
                predicted_class_name,
            )
            log_cuda_memory(self.logger, f"BEFORE BOX {prediction_num}")

            # Clear cache before each box processing
            _empty_cuda_cache()

            # CRP visualization
            # explanation_fig = plot_one_image_explanation(
            #     model_name, self.person_vehicle_model, image_tensor,
            #     self.person_car_dataset, class_id, layer, prediction_num,
            #     mode, n_concepts, n_refimgs, output_dir=glocal_analysis_output_dir
            # )

            # PCX visualization
            try:
                if sample_id is not None:
                    explanation_fig = plot_pcx_explanations(
                        class_id=class_id,
                        model_name=model_name,
                        model=self.person_vehicle_model,
                        dataset=self.person_car_crp_dataset,
                        sample_id=sample_id,
                        n_concepts=n_concepts,
                        n_refimgs=n_refimgs,
                        num_prototypes=prototype_dict,
                        prediction_num=class_local_prediction_num,
                        layer_name=layer,
                        ref_imgs_path=ref_imgs_path,
                        output_dir_pcx=pcx_output_dir,
                        output_dir_crp=crp_output_dir,
                        display_class_names=display_class_names,
                    )
                else:
                    explanation_fig = plot_one_image_pcx_explanation(
                        model_name=model_name,
                        model=self.person_vehicle_model,
                        img=image_tensor,
                        orig_img=image,
                        dataset=self.person_car_crp_dataset,
                        orig_dataset=self.person_car_dataset_orig,
                        class_id=class_id,
                        n_concepts=n_concepts,
                        n_refimgs=n_refimgs,
                        num_prototypes=prototype_dict,
                        prediction_num=class_local_prediction_num,
                        layer_name=layer,
                        ref_imgs_path=ref_imgs_path,
                        output_dir_pcx=pcx_output_dir,
                        output_dir_crp=crp_output_dir,
                        outside_logger=self.logger,
                        display_class_names=display_class_names,
                    )
            except ExplanationUnavailableError as exc:
                self.logger.info(
                    "Skipping explanation for detection %s (%s, confidence %.3f): %s",
                    prediction_num,
                    predicted_class_name,
                    confidence,
                    exc,
                )
                gc.collect()
                _empty_cuda_cache()
                continue
            figure_kpis = getattr(explanation_fig, "_kpi_metrics", {})
            backward_time_s = float(figure_kpis.get("backward_time_s", 0.0))
            full_attribution_time_s = float(figure_kpis.get("full_attribution_time_s", 0.0))
            box_backward_times.append(backward_time_s)
            box_full_times.append(full_attribution_time_s)

            yolo_box_record = build_kpi_record(
                model="yolo",
                entity_type=original_entity_type,
                scope="box",
                aggregation="raw",
                image=original_image_filename,
                box_index=prediction_num,
                num_boxes=num_boxes,
                class_id=class_id,
                confidence=f"{confidence:.6f}",
                bbox=boxes_list[prediction_num],
                layer=layer,
                n_concepts=n_concepts,
                n_refimgs=n_refimgs,
                prediction_time_s=prediction_time_s,
                global_lcrp_time_s=backward_time_s,
                global_total_time_s=full_attribution_time_s,
            )
            self._append_kpi_record_all("yolo_box_kpis.txt", yolo_box_record)
            self._append_avg_window_record_all(
                source_filename="yolo_box_kpis.txt",
                avg_filename="yolo_box_kpis_avg5.txt",
                model="yolo",
                entity_type=original_entity_type,
                scope="box",
                layer=layer,
            )

            explanation_img = fig_to_array(explanation_fig)
            plt.close(explanation_fig)
            explanation_images.append(explanation_img)

            explanation_file_name = f"tfa02/{original_entity_type}/{original_filename_no_ext}/object_{prediction_num}.png"
            explanation_image_filenames.append(explanation_file_name)
            explanation_boxes.append({
                "object_id": prediction_num,
                "bbox": boxes_list[prediction_num],
                "class_id": class_id,
                "confidences": confidence,
                "explanation_image": explanation_file_name,
                "explanation_image_bucket": FHHI_MINIO_BUCKET,
            })

            log_cuda_memory(self.logger, f"AFTER BOX {prediction_num}")

            # Force garbage collection after each box
            gc.collect()
            _empty_cuda_cache()

        total_backward_time_s = sum(box_backward_times)
        total_full_attribution_time_s = sum(box_full_times)
        attribution_single_time_s = mean(box_full_times) if box_full_times else 0.0
        yolo_image_record = build_kpi_record(
            model="yolo",
            entity_type=original_entity_type,
            scope="image",
            aggregation="raw",
            image=original_image_filename,
            num_boxes=num_boxes,
            layer=layer,
            n_concepts=n_concepts,
            n_refimgs=n_refimgs,
            prediction_time_s=prediction_time_s,
            global_lcrp_time_s=total_backward_time_s,
            global_total_time_s=total_full_attribution_time_s,
            attribution_single_time_s=attribution_single_time_s,
        )
        yolo_image_logs = self._append_kpi_record_all("yolo_image_kpis.txt", yolo_image_record)
        _, yolo_avg5_logs = self._append_avg_window_record_all(
            source_filename="yolo_image_kpis.txt",
            avg_filename="yolo_image_kpis_avg5.txt",
            model="yolo",
            entity_type=original_entity_type,
            scope="image",
            layer=layer,
        )
        self.latest_kpi_log_paths = (
            self._log_path_variants("yolo_box_kpis.txt")
            + self._log_path_variants("yolo_box_kpis_avg5.txt")
            + yolo_image_logs
            + yolo_avg5_logs
        )

        log_cuda_memory(self.logger, "PERSON_VEHICLE END")
        _empty_cuda_cache()

        explanation_entity = get_person_vehicle_detection_explanation_entity(
            original_image_bucket=original_image_bucket,
            original_image_filename=original_image_filename,
            original_detection_boxes=boxes_list,
            original_detection_class_categories=class_ids.detach().cpu().tolist(),
            original_detection_confidences=confidences.detach().cpu().tolist(),
            explanation_boxes=explanation_boxes,
            n_concepts=n_concepts,
            n_refimgs=n_refimgs,
            layer=layer,
            mode=mode,
            bm_id=bm_id,
            uav_id=uav_id,
            flight_number=flight_number,
            alert_ref=alert_ref,
        )
        return explanation_entity, explanation_images, explanation_image_filenames

    def explain_smoke_segmentation(self, src_entity: dict, image: np.ndarray):
        raise NotImplementedError("Smoke segmentation explanation is not implemented yet.")


# Configure basic logging if not done elsewhere
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Resolve to the repository/application root both locally and in Docker.
# PROJECT_ROOT can override this when code and runtime assets are mounted at
# different locations.
project_root = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
)
logger.info(f"Project Root set to: {project_root}")

# THIS IS WHERE THE 'explanator' OBJECT IS CREATED
explanator = Explanator(project_root=project_root, logger=logger)
logger.info("Explanator initialized successfully.")
