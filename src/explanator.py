import os
import torch
import torchvision.transforms as transforms
import numpy as np

from LCRP.models import get_model 
from src.plot_crp_explanations import plot_one_image_explanation, fig_to_array
from src.datasets.person_car_dataset import PersonCarDataset


class Explanator:
    """Class that stores all loaded models together with all relevant data for generating CRP explanations.
    
    This is the main class used in the TFA-02 component. 
    """
    
    def __init__(self, project_root: str):
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
        }

        self.VALID_ENTITY_TYPES = list(self.entity_handlers.keys())
    
    def explain(self, entity_type: str, image):
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
        return handler(image)
    
    def explain_burnt_segmentation(self, image):
        # Implementation for burnt segmentation explanation
        pass
    

    def explain_fire_segmentation(self, image):
        # Implementation for fire segmentation explanation
        pass
    

    def explain_flood_segmentation(self, image):
        # Implementation for flood segmentation explanation
        pass
    

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

    def explain_person_vehicle_detection(self, image: np.ndarray):
        # Implementation for person/vehicle detection explanation

        model_name = "yolov6s6"

        # Setting up main parameters
        class_id = 1
        n_concepts = 3
        n_refimgs = 12
        layer = "module.backbone.ERBlock_6.2.cspsppf.cv7.block.conv"
        mode = "relevance"
        prediction_num = 0

        glocal_analysis_output_dir = "output/crp/yolo_person_car"

        # turn image into tensor
        image_tensor = torch.tensor(image, dtype=self.dtype)


        explanation_fig = plot_one_image_explanation(model_name, self.person_vehicle_model, image_tensor, self.person_car_dataset, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir=glocal_analysis_output_dir)
        explanation_img = fig_to_array(explanation_fig)
        return {"entity": "TO BE IMPLEMENTED"}, explanation_img

    
    

    def explain_smoke_segmentation(self, image):
        # Implementation for smoke segmentation explanation
        pass