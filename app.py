from flask import Flask, request, jsonify, render_template, g
import requests
import os
import logging
import numpy as np
from PIL import Image

from src.explanator import Explanator
from src.minio_client import MinIOClient, MINIO_BUCKET

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

app = Flask(__name__, template_folder='./')

# Configure logging, set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Get base path from environment variable
BASE_PATH = os.environ.get('BASE_PATH', '/')


def get_minio_client():
    if 'minio_client' not in g:
        g.minio_client = MinIOClient()
    return g.minio_client


def get_explanator():
    if 'explanator' not in g:
        g.explanator = Explanator(project_root=PROJECT_ROOT)
    return g.explanator


# Route for the index page
@app.route(f'{BASE_PATH}')
def index():
    return render_template('index.html', message='Welcome! This demonstration container illustrates how to integrate services. Feel free to explore and experiment.')


# Route for the ping endpoint
@app.route(f'{BASE_PATH}/ping')
def ping(): 
    app.logger.debug('Ping endpoint called.')
    return jsonify({'status': 'ok'})


# Route for the GET request
@app.route(f'{BASE_PATH}/get_data', methods=['GET'])
def get_data(): 
    try: 
        app.logger.debug('GET request received.')
        # Logic to retrieve data would go here
        return jsonify({'message': 'Welcome! This is a sample service.'}) 
    except Exception as e:  
        app.logger.error(f'Error in GET request: {str(e)}')  
        return jsonify({'error': str(e)}), 500


# Route for the POST request
@app.route(f'{BASE_PATH}/post_data', methods=['POST']) 
def post_data():
    try: 
        app.logger.debug('POST request received.') 
        data = request.get_json() 

        # AUTH entities are a list of one element
        data = data[0]

        entity_type = data.get('type')

        if entity_type is None:
            return jsonify({'error': 'Entity type not provided.'}), 400
        
        explanator = get_explanator()

        if entity_type not in explanator.VALID_ENTITY_TYPES:
            return jsonify({'error': f'Invalid entity type: {entity_type}.'}), 400

        # Load the input image from MinIO
        filename = data["parameters"]["value"]["FileName"]

        minio_filename = f"{filename}"

        minio_client = get_minio_client()

        img = minio_client.download_image(MINIO_BUCKET, minio_filename)
        if img is None:
            return jsonify({'error': f'Error downloading image from MinIO: {minio_filename}'}), 500


        explanation_entity, explanation_image = explanator.explain(entity_type, img)
        
        # Upload the image to MinIO
        # explanation_image_filename = f"EXPLANATION/{entity_type}/{filename}"
        # minio_client.upload_image(MINIO_BUCKET, explanation_image_filename, explanation_image)

        # Send the explanation entity to the Context Broker
        # entity_id = # TODO: Implement this
        # entity_type = # TODO: Implement this
        # create_entity_response = create_entity(entity_id, entity_type, explanation_entity)

        print(np.asarray(Image.fromarray(explanation_image)).shape)
        explanation_image = Image.fromarray(explanation_image)
        explanation_image.show()

        print(np.asarray(Image.fromarray(img)).shape)
        img = Image.fromarray(img)
        img.show()

        return jsonify({'message': 'Explanation successful', 'explanation_entity': explanation_entity}), 200
        

        if 'name' in data: 
            message = f"Hello, {data['name']}!"
            return jsonify({'message': message}) 
        return jsonify({'message': data}) 
    except Exception as e:
        app.logger.error(f'Error in POST request: {str(e)}') 
        return jsonify({'error': str(e)}), 500


# Endpoint to send data to the Context Broker
@app.route(f'{BASE_PATH}/send_to_context_broker', methods=['POST']) 
def send_to_context_broker(): 
    try:
        app.logger.debug('POST request to send data to context broker received.') 
        data = request.get_json() 
        if not data: 
            raise ValueError('No data provided in the request.')
        value = data.get('value')  
        entity_id = "urn:ngsi-ld:" + os.environ.get('BROKER_ENTITY_ID')
        entity_type = os.environ.get('BROKER_TYPE_ID')
        create_entity_response = create_entity(entity_id, entity_type, value)

        if create_entity_response.status_code == 201:
            return {'message': f'Entity created successfully: {entity_id}'}, 201
        if create_entity_response.status_code != 409:
            return {'error': f'Error creating entity: {create_entity_response.text}'}, create_entity_response.status_code

        data_to_send = {
            "title": {"type": "Property", "value": "PropertyValue"},
            "description": {"type": "Property", "value": "Value of the description field"},
            "additionalAttribute": {"type": "Property", "value": value}
        }

        update_attrs_url = f'{os.environ.get("BROKER_URL")}/ngsi-ld/v1/entities/{entity_id}/attrs'
        response = requests.post(update_attrs_url, json=data_to_send)

        status_map = {
            201: {'log_message': 'Data successfully sent to Orion.', 'json_response': {'message': 'Data successfully sent to Orion.'}},
            204: {'log_message': 'Data successfully sent to Orion.', 'json_response': {'message': 'Data successfully sent to Orion.'}},
            400: {'log_message': 'Bad request to Orion.', 'json_response': {'error': 'Bad request to Orion'}},
            401: {'log_message': 'Unauthorized request to Orion.', 'json_response': {'error': 'Unauthorized request to Orion'}},
            404: {'log_message': 'Resource not found on Orion.', 'json_response': {'error': 'Resource not found on Orion'}},
            'default': {'log_message': 'Unexpected response from Orion.', 'json_response': {'error': 'Unexpected response from Orion'}}
        }

        status_info = status_map.get(response.status_code, status_map['default'])
        log_message = status_info['log_message']
        json_response = status_info['json_response']
        app.logger.debug(log_message)
        return jsonify(json_response), response.status_code  

    except Exception as e: 
        app.logger.error(f'Error while sending data to Orion: {str(e)}') 
        return jsonify({'error': f'Error while sending data to Orion: {str(e)}'}), 500 

def create_entity(entity_id, entity_type, value):
    raise NotImplementedError("Implement this for our TFA02 entities")
    data_to_send = {
        "id": entity_id,
        "type": entity_type,
        "creator": "Team_name",
        "title": {"type": "Property", "value": "PropertyValue"},
        "description": {"type": "Property", "value": "Value of the description field"},
        "additionalAttribute": {"type": "Property", "value": {"name": "Attribute", "value": value}}
    }
    create_entity_url = f'{os.environ.get("BROKER_URL")}/ngsi-ld/v1/entities'
    response = requests.post(create_entity_url, json=data_to_send)
    return response

# Run the Flask application
if __name__ == '__main__':
    debug = os.environ.get('DEBUG', '').lower() in ('true', '1')
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT')), debug=debug)