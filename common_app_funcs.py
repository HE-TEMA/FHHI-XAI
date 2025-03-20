import json
import os
import redis
from rq import Queue
import requests

# Using Redis as a simple database
def set_bm_id(redis_conn, bm_id):
    redis_conn.set('current_bm_id', bm_id)

def get_bm_id(redis_conn):
    bm_id = redis_conn.get('current_bm_id')
    return bm_id.decode('utf-8') if bm_id else None

def update_job_status(redis_conn, task_id, status_data):
    redis_conn.set(f"job_status:{task_id}", json.dumps(status_data))

def get_job_status(redis_conn, task_id):
    status_json = redis_conn.get(f"job_status:{task_id}")
    if status_json:
        return json.loads(status_json)
    return None

def get_redis_conn():
    return redis.Redis(
    host='localhost',  # Using localhost since Redis runs in same container
    port=int(os.environ.get('REDIS_PORT', 6379)),
    password=None,
    db=0
)

def get_job_queue(redis_conn):
    return Queue('image_processing', connection=redis_conn)



def update_entity(entity_to_send):
    update_entity_url = "https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/entityOperations/upsert"
    response = requests.post(update_entity_url, json=entity_to_send)
    return response