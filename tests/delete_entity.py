import argparse
import json
import requests

from entity_paths import entity_type_to_entity_path, entity_type_to_entity_id



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("entity_type", type=str, help="Entity type to test", choices=entity_type_to_entity_path.keys())

    args = parser.parse_args()

    entity_type = args.entity_type

    entity_id =  entity_type_to_entity_id[entity_type]
    if entity_id == "":
        raise ValueError(f"Entity ID for {entity_type} is not defined.")
     
    entity_data = {
        "entity_id": entity_id,
    }
    hostname = "http://localhost"
    port = 4000
    endpoint = "explanation/delete_entity"
    url = f"{hostname}:{port}/{endpoint}"

    response = requests.post(url, json=entity_data)

    try:
        print(response.json())
    except:
        print(f"Failed to get response as a json")
        print(response.text)

