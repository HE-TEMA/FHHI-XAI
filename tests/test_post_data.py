import argparse
import json
import requests

from entity_paths import entity_type_to_entity_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("entity_type", type=str, help="Entity type to test", choices=entity_type_to_entity_path.keys())

    args = parser.parse_args()

    entity_type = args.entity_type
     
    entity_path = entity_type_to_entity_path[entity_type]

    with open(f"{entity_path}") as f:
        entity_data = json.load(f)

    hostname = "http://localhost"
    port = 14
    endpoint = "explanation/post_data"
    url = f"{hostname}:{port}/{endpoint}"

    response = requests.post(url, json=entity_data)

    print(response.json())

