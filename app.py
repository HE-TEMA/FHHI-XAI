from flask import Flask, request, jsonify, render_template, g
import requests
import os
import logging
import numpy as np
import traceback
from PIL import Image

from src.explanator import Explanator
from src.minio_client import MinIOClient, FHHI_MINIO_BUCKET, NAPLES_MINIO_BUCKET

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

app = Flask(__name__, template_folder='./')

# Configure logging, set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Configure logging to silence specific libraries
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
        raw_data = request.get_json() 

        # AUTH entities are a list of one element
        entity = raw_data[0]

        entity_type = entity.get('type')

        if entity_type is None:
            return jsonify({'error': 'Entity type not provided.'}), 400
        
        explanator = get_explanator()

        if entity_type not in explanator.VALID_ENTITY_TYPES:
            return jsonify({'error': f'Invalid entity type: {entity_type}.'}), 400

        # Load the input image from MinIO
        filename = entity["parameters"]["value"]["FileName"]

        minio_filename = f"{filename}"

        minio_client = get_minio_client()

        img = minio_client.download_image(NAPLES_MINIO_BUCKET, minio_filename)
        if img is None:
            return jsonify({'error': f'Error downloading image from MinIO: {minio_filename}'}), 500


        explanation_entity, explanation_image = explanator.explain(entity_type, entity, img)
        
        app.logger.info("Uploading explanation image to MinIO")
        explanation_image_filename = explanation_entity[0]["explanation_image"]["value"]["filename"]
        minio_client.upload_image(FHHI_MINIO_BUCKET, explanation_image_filename, explanation_image)

        app.logger.info("Sending explanation entity to Orion")
        orion_response = update_entity(explanation_entity)

        status_map = {
            204: {'json_response': {'message': 'Entity successfully updated on Orion.'}},
            400: {'json_response': {'error': 'Bad request to Orion'}},
            401: {'json_response': {'error': 'Unauthorized request to Orion'}},
            404: {'json_response': {'error': 'Entity not found on Orion'}},
            'default': {'json_response': {'error': 'Unexpected response from Orion'}}
        }

        status_info = status_map.get(orion_response.status_code, status_map['default'])
        orion_json_response = status_info['json_response']

        if orion_response.status_code != 204:
            return jsonify(orion_json_response), orion_response.status_code 


        print(np.asarray(Image.fromarray(explanation_image)).shape)
        explanation_image = Image.fromarray(explanation_image)
        explanation_image.show()

        print(np.asarray(Image.fromarray(img)).shape)
        img = Image.fromarray(img)
        img.show()

        return jsonify({'message': 'Explanation successful', 'explanation_entity': explanation_entity, 'orion_response': orion_json_response}), 200
        
    except Exception as e:
        app.logger.error(f'Unexpected error: {str(e)}')
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}, traceback: {traceback.format_exc()}'}), 500



def update_entity(entity_to_send):
    update_entity_url = "https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/entityOperations/upsert"
    response = requests.post(update_entity_url, json=entity_to_send)
    return response


@app.route(f'{BASE_PATH}/delete_entity', methods=['POST'])
def delete_entity():
    try:
        app.logger.debug('POST request to delete entity received.')
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided in the request.'}), 400
        
        entity_id = data.get('entity_id')

        if entity_id is None:
            return jsonify({'error': 'Entity ID not provided.'}), 400
            
        # Construct the delete URL
        broker_url = os.environ.get('BROKER_URL', 'https://orion.tema.digital-enabler.eng.it')
        delete_entity_url = f"{broker_url}/ngsi-ld/v1/entities/{entity_id}"
        
        # Send the delete request to the Orion Context Broker
        app.logger.debug(f'Sending DELETE request to: {delete_entity_url}')
        response = requests.delete(delete_entity_url)
        
        status_map = {
            204: {'log_message': 'Entity successfully deleted from Orion.', 'json_response': {'message': 'Entity successfully deleted.'}},
            400: {'log_message': 'Bad request to Orion.', 'json_response': {'error': 'Bad request to Orion'}},
            401: {'log_message': 'Unauthorized request to Orion.', 'json_response': {'error': 'Unauthorized request to Orion'}},
            404: {'log_message': 'Entity not found on Orion.', 'json_response': {'error': 'Entity not found on Orion'}},
            'default': {'log_message': 'Unexpected response from Orion.', 'json_response': {'error': 'Unexpected response from Orion'}}
        }
        
        status_info = status_map.get(response.status_code, status_map['default'])
        log_message = status_info['log_message']
        json_response = status_info['json_response']
        
        app.logger.debug(f'{log_message} Status code: {response.status_code}')
        return jsonify(json_response), response.status_code
        
    except Exception as e:
        app.logger.error(f'Error in delete entity request: {str(e)}')
        return jsonify({'error': str(e)}), 500


@app.route(f'{BASE_PATH}/subscribe_to_context_broker', methods=['POST'])
def subscribe_to_context_broker():
    try:
        app.logger.debug('POST request to subscribe to context broker received.')
        
        # Delete existing subscription if it exists
        subscription_id = "urn:ngsi-ld:fhhi:one_subscription_to_rule_them_all"

        broker_url = os.environ.get('BROKER_URL', 'https://orion.tema.digital-enabler.eng.it')
        delete_url = f"{broker_url}/ngsi-ld/v1/subscriptions/{subscription_id}"
        
        app.logger.debug(f'Attempting to delete existing subscription: {delete_url}')
        delete_response = requests.delete(delete_url)
        app.logger.debug(f'Delete subscription response: {delete_response.status_code}')
        
        # Get the notification endpoint
        host = request.host_url.rstrip('/')
        post_data_url = f"{host}{BASE_PATH}/post_data"
        app.logger.info(f'POST_DATA_URL: {post_data_url}')
        
        # Create subscription payload
        subscription_payload = {
            "id": subscription_id,
            "type": "Subscription",
            "name": "TFA02 inputs subscription",
            "description": "Subscription description",
            "entities": [
                {"type": "BurntSegmentation"},
                {"type": "FireSegmentation"},
                {"type": "FloodSegmentation"},
                {"type": "PersonVehicleDetection"},
                {"type": "SmokeSegmentation"}
            ],
            "notification": {
                "endpoint": {
                    "uri": post_data_url,
                    "accept": "application/json"
                }
            }
        }
        
        # Create new subscription
        subscription_url = f"{broker_url}/ngsi-ld/v1/subscriptions"
        app.logger.debug(f'Creating subscription at: {subscription_url}')
        
        create_response = requests.post(
            subscription_url,
            headers={'Content-Type': 'application/json'},
            json=subscription_payload
        )
        
        status_map = {
            201: {'message': 'Subscription created successfully.'},
            204: {'message': 'Subscription updated successfully.'},
            400: {'error': 'Bad request to Orion Context Broker.'},
            401: {'error': 'Unauthorized request to Orion Context Broker.'},
            403: {'error': 'Forbidden request to Orion Context Broker.'},
            422: {'error': 'Unprocessable Entity - Invalid subscription payload.'},
            'default': {'error': f'Unexpected response from Orion Context Broker: {create_response.status_code}'}
        }
        
        response_info = status_map.get(create_response.status_code, status_map['default'])
        app.logger.info(f'Subscription response: {create_response.status_code} - {response_info}')
        
        if create_response.status_code in [201, 204]:
            return jsonify(response_info), create_response.status_code
        else:
            return jsonify(response_info), create_response.status_code

    except Exception as e:
        app.logger.error(f'Error in subscribe to context broker request: {str(e)}')
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Endpoint to send data to the Context Broker
@app.route(f'{BASE_PATH}/send_to_context_broker', methods=['POST']) 
def send_to_context_broker(): 
    raise NotImplementedError("Implement this for our TFA02 entities")

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