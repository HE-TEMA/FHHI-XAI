import copy
from datetime import datetime

person_vehicle_detection_explanation_template = {
    "id": "urn:ngsi-ld:tema:FHHI:TFA-02:PersonVehicleDetectionExplanation:fhhi-tfa02_auth-tfa05_001",
    "type": "PersonVehicleDetectionExplanation",
    "creator": {
        "type": "Property",
        "value": "FHHI (TFA-02)"
    },
    "title": {
        "type": "Property",
        "value": "Person Vehicle Detection Explanation."
    },
    "description": {
        "type": "Property",
        "value": "Explanation created by FHHI (TFA-02) for AUTH (TFA-05) related to person/vehicle detection."
    },
    "bucket": {
        "type": "Property",
        "value": "fhhi"
    },
    "explainsEntity": {
        "type": "Relationship",
        "object": "urn:ngsi-ld:tema:AUTH:TFA-05:PersonVehicleDetection"
    },
    "targetComponent": {
        "type": "Property",
        "value": "AUTH (TFA-05)"
    },
    "timestamp": {
        "type": "Property",
        "value": None
    },

    # Comes from the original detection entity
    "original_image": {
        "type": "Property",
        "value": {
            "bucket": None,
            "filename": None,
        },
    },
    # Comes from the original detection entity
    "original_detection": {
        "type": "Property",
        "value": {
            "boxes": None,
            "class_categories": None,
        },
    },
    "explanation": {
        "type": "Property",
        "value": {
            "boxes": None,
            "parameters": {
                "n_concepts": None,
                "n_refimgs": None,
                "layer": None,
                "mode": None,
            }
        }
    },
}

def get_person_vehicle_detection_explanation_entity(
    original_image_bucket,
    original_image_filename,
    original_detection_boxes,
    original_detection_class_categories,
    explanation_boxes,
    n_concepts,
    n_refimgs,
    layer,
    mode,
):
    template = copy.deepcopy(person_vehicle_detection_explanation_template)

    timestamp = datetime.now().isoformat()

    template["timestamp"]["value"] = timestamp
    template["original_image"]["value"]["bucket"] = original_image_bucket
    template["original_image"]["value"]["filename"] = original_image_filename
    template["original_detection"]["value"]["boxes"] = original_detection_boxes
    template["original_detection"]["value"]["class_categories"] = original_detection_class_categories
    template["explanation"]["value"]["boxes"] = explanation_boxes
    template["explanation"]["value"]["parameters"]["n_concepts"] = n_concepts
    template["explanation"]["value"]["parameters"]["n_refimgs"] = n_refimgs
    template["explanation"]["value"]["parameters"]["layer"] = layer
    template["explanation"]["value"]["parameters"]["mode"] = mode

    # For some reason we need to upload a list of one element, according to AUTH example script
    return [template]


flood_segmentation_explanation_template = {
    "id": "urn:ngsi-ld:tema:FHHI:TFA-02:FloodSegmentationExplanation:fhhi-tfa02_auth-tfa06_001",
    "type": "FloodSegmentationExplanation",
    "creator": {
        "type": "Property",
        "value": "FHHI (TFA-02)"
    },
    "title": {
        "type": "Property",
        "value": "Flood Segmentation Explanation."
    },
    "description": {
        "type": "Property",
        "value": "Explanation created by FHHI (TFA-02) for AUTH (TFA-06) related to flood segmentation."
    },
    "bucket": {
        "type": "Property",
        "value": "fhhi"
    },
    "explainsEntity": {
        "type": "Relationship",
        "object": "urn:ngsi-ld:tema:AUTH:TFA-06:FloodSegmentation"
    },
    "targetComponent": {
        "type": "Property",
        "value": "AUTH (TFA-06)"
    },
    "timestamp": {
        "type": "Property",
        "value": None
    },

    # Comes from the original segmentation entity
    "original_image": {
        "type": "Property",
        "value": {
            "bucket": None,
            "filename": None,
        },
    },
    # Comes from the original segmentation entity
    "original_segmentation": {
        "type": "Property",
        "value": {
            "mask_id": None,
        },
    },
    "explanation_image": {
        "type": "Property",
        "value": {
            "bucket": None,
            "filename": None,
        },
    },
    "explanation": {
        "type": "Property",
        "value": {
            "class_id": None,
            "n_concepts": None,
            "n_refimgs": None,
            "layer": None,
            "mode": None,
        }
    },
}

def get_flood_segmentation_explanation_entity(
    original_image_bucket,
    original_image_filename,
    original_segmentation_mask_id,
    explanation_image_bucket,
    explanation_image_filename,
    class_id,
    n_concepts,
    n_refimgs,
    layer,
    mode,
):
    template = copy.deepcopy(flood_segmentation_explanation_template)

    timestamp = datetime.now().isoformat()

    template["timestamp"]["value"] = timestamp
    template["original_image"]["value"]["bucket"] = original_image_bucket
    template["original_image"]["value"]["filename"] = original_image_filename
    template["original_segmentation"]["value"]["mask_id"] = original_segmentation_mask_id
    template["explanation_image"]["value"]["bucket"] = explanation_image_bucket
    template["explanation_image"]["value"]["filename"] = explanation_image_filename
    template["explanation"]["value"]["class_id"] = class_id
    template["explanation"]["value"]["n_concepts"] = n_concepts
    template["explanation"]["value"]["n_refimgs"] = n_refimgs
    template["explanation"]["value"]["layer"] = layer
    template["explanation"]["value"]["mode"] = mode

    # For some reason we need to upload a list of one element, according to AUTH example script
    return [template]