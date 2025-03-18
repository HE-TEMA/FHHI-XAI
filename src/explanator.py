import os
import torch
import torchvision.transforms as transforms
import numpy as np
# Set non-interactive backend for matplotlib to avoid GUI issues in Flask
import matplotlib
matplotlib.use('Agg')
import copy
import logging

from LCRP.models import get_model 
from src.plot_crp_explanations import plot_one_image_explanation, fig_to_array
from src.datasets.person_car_dataset import PersonCarDataset
from src.datasets.flood_dataset import FloodDataset
from src.entities import get_person_vehicle_detection_explanation_entity, get_flood_segmentation_explanation_entity
from src.minio_client import FHHI_MINIO_BUCKET


class Explanator:
    """Class that stores all loaded models together with all relevant data for generating CRP explanations.
    
    This is the main class used in the TFA-02 component. 
    """
    
    def __init__(self, project_root: str, logger: logging.Logger):
        self.logger = logger
        # General setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        self.project_root = project_root

        self.person_vehicle_model = self.load_person_vehicle_model()
        self.person_car_dataset = self.load_person_car_data()

        # Create a mapping from entity types to handler methods
        self.entity_handlers = {
            "BurntSegmentation": self.explain_burnt_segmentation,
            "FireSegmentation": self.explain_fire_segmentation,
            "FloodSegmentation": self.explain_flood_segmentation,
            "PersonVehicleDetection": self.explain_person_vehicle_detection,
            "SmokeSegmentation": self.explain_smoke_segmentation,
            "EOBurntArea": self.explain_eo_burnt_area,
            "EOFloodExtent": self.explain_eo_flood_extent,
        }

        self.VALID_ENTITY_TYPES = list(self.entity_handlers.keys())
        self.DLR_ENTITY_TYPES = {"EOBurntArea", "EOFloodExtent"}
    
    def explain(self, entity_type: str, src_entity: dict, image: np.ndarray):
        """Generate explanation for the given entity type and image.
        
        Args:
            entity_type: Type of entity to explain (must be in VALID_ENTITY_TYPES)
            image: Input to the corresponding model. This was also input to the entity that did Detection/Segmentation,
                but for Explanations we need to run the model once again.
            
        Returns:
            dict, np.ndarray: Explanation entity and explanation image
        """
        if entity_type not in self.VALID_ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {entity_type}. Must be one of {self.VALID_ENTITY_TYPES}")
        
        # Get the appropriate handler method for this entity type
        handler = self.entity_handlers.get(entity_type)
        
        # Call the handler method with the image
        return handler(src_entity, image)

    def explain_eo_burnt_area(self, src_entity: dict, image: np.ndarray):
        raise NotImplementedError("EO Burnt Area explanation is not implemented yet.")
    
    def explain_eo_flood_extent(self, src_entity: dict, image: np.ndarray):
        raise NotImplementedError("EO Flood Extent explanation is not implemented yet.")
    
    def explain_burnt_segmentation(self, src_entity: dict, image: np.ndarray): 
        raise NotImplementedError("Burnt segmentation explanation is not implemented yet.")
    

    def explain_fire_segmentation(self, src_entity: dict, image: np.ndarray):
        raise NotImplementedError("Fire segmentation explanation is not implemented yet.")
    


    def explain_flood_segmentation(self, src_entity: dict, image: np.ndarray):
        """
        Generate flood segmentation explanation using UNet model and CRP method.
        
        Args:
            src_entity: Source entity with flood segmentation data
            image: Input image as numpy array
            
        Returns:
            Tuple of (explanation_entity, explanation_image)
        """
        # Load the flood segmentation model
        model_name = "unet"
        flood_model_path = os.path.join(self.project_root, "models", "unet_flood_modified.pt")
        
        # Load model if not already loaded
        if not hasattr(self, "flood_model"):
            self.flood_model = get_model(model_name=model_name, classes=2, ckpt_path=flood_model_path, device=self.device, dtype=self.dtype)
        
        # Load flood dataset if not already loaded
        if not hasattr(self, "flood_dataset"):
            flood_data_path = os.path.join(self.project_root, "data", "General_Flood_v3")
            
            # Define the transform for flood dataset
            transform = transforms.Compose([
                transforms.ToTensor(),  # Convert to tensor
                transforms.Lambda(lambda x: x.to(self.dtype)),
            ])
            
            self.flood_dataset = FloodDataset(root_dir=flood_data_path, split="train", transform=transform)
        
        # Setting up main parameters for explanation
        class_id = 1  # Flood class ID
        n_concepts = 3
        n_refimgs = 12
        layer = "encoder.features.15"  # Based on UNet example
        mode = "relevance"
        prediction_num = 0
        
        glocal_analysis_output_dir = "output/crp/unet_flood"
        
        # Apply all transform to the input test image. They are aplied here separately, but normally this is just done in the get_item method of the dataset.
        # This is done to be consistent with the reference images.
        image_tensor = self.flood_dataset.transform(image)
        image_tensor = self.flood_dataset.resize(image_tensor)
        
        # Generate explanation
        explanation_fig = plot_one_image_explanation(
            model_name, self.flood_model, image_tensor, self.flood_dataset, 
            class_id, layer, prediction_num, mode, n_concepts, n_refimgs, 
            output_dir=glocal_analysis_output_dir
        )
        explanation_img = fig_to_array(explanation_fig)
        
        # Prepare explanation entity
        original_filename = src_entity["parameters"]["value"]["FileName"]
        original_entity_type = src_entity["type"]
        
        explanation_image_filename = f"tfa02/{original_entity_type}/{original_filename}"
        
        explanation_entity = get_flood_segmentation_explanation_entity(
            original_image_bucket=src_entity["bucket"]["value"],
            original_image_filename=original_filename,
            original_segmentation_mask_id=src_entity["segmentation"]["value"]["mask_id"],
            explanation_image_bucket=FHHI_MINIO_BUCKET,
            explanation_image_filename=explanation_image_filename,
            class_id=class_id,
            n_concepts=n_concepts,
            n_refimgs=n_refimgs,
            layer=layer,
            mode=mode
        )

        return explanation_entity, [explanation_img], [explanation_image_filename]
    

    def load_person_vehicle_model(self):
        # Load the person/vehicle detection model
        model_name = "yolov6s6"
        person_vehicle_model_path = os.path.join(self.project_root, "models" , "best_v6s6_ckpt.pt")
        return get_model(model_name=model_name, classes=2, ckpt_path=person_vehicle_model_path, device=self.device, dtype=self.dtype)

    def load_person_car_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Resize((1280, 1280)),
            transforms.Lambda(lambda x: x.to(self.dtype)), 
        ])

        person_car_data_path = os.path.join(self.project_root, "data" , "person_car_detection_data", "Arthal")
        dataset = PersonCarDataset(root_dir=person_car_data_path, split="train", transform=transform)
        return dataset

    def explain_person_vehicle_detection(self, src_entity: dict, image: np.ndarray):
        # Implementation for person/vehicle detection explanation

        model_name = "yolov6s6"

        # Setting up main parameters
        n_concepts = 3
        n_refimgs = 12
        layer = "module.backbone.ERBlock_6.2.cspsppf.cv7.block.conv"
        mode = "relevance"

        glocal_analysis_output_dir = "output/crp/yolo_person_car"

        original_predicted_boxes = src_entity["detection"]["value"]["boxes"]
        original_filename = src_entity["parameters"]["value"]["FileName"]
        original_filename_no_ext = os.path.splitext(original_filename)[0]
        original_entity_type = src_entity["type"]

        # Apply transform as with the reference images
        image_tensor = self.person_car_dataset.transform(image)

        explanation_boxes = copy.deepcopy(original_predicted_boxes)
        num_boxes = len(explanation_boxes)
        explanation_images = []
        explanation_image_filenames = []
        for prediction_num in range(num_boxes):
            self.logger.debug(f"Generating explanation for box {prediction_num} of {num_boxes}")

            class_id = explanation_boxes[prediction_num]["category_id"]
            explanation_fig = plot_one_image_explanation(model_name, self.person_vehicle_model, image_tensor, self.person_car_dataset, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir=glocal_analysis_output_dir)
            explanation_img = fig_to_array(explanation_fig)
            explanation_images.append(explanation_img)

            explanation_file_name = f"tfa02/{original_entity_type}/{original_filename_no_ext}/object_{prediction_num}.png"
            explanation_image_filenames.append(explanation_file_name)

            explanation_boxes[prediction_num]["explanation_image"] = explanation_file_name
            explanation_boxes[prediction_num]["explanation_image_bucket"] = FHHI_MINIO_BUCKET



        explanation_entity = get_person_vehicle_detection_explanation_entity(
            original_image_bucket=src_entity["bucket"]["value"],
            original_image_filename=original_filename,
            original_detection_boxes=original_predicted_boxes,
            original_detection_class_categories=src_entity["detection"]["value"]["class_categories"],
            explanation_boxes=explanation_boxes,
            n_concepts=n_concepts,
            n_refimgs=n_refimgs,
            layer=layer,
            mode=mode,
        )

        return explanation_entity, explanation_images, explanation_image_filenames

    
    
    def explain_smoke_segmentation(self, src_entity: dict, image: np.ndarray):
        raise NotImplementedError("Smoke segmentation explanation is not implemented yet.")