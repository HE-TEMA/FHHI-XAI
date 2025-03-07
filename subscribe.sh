# Command to subscribe to make a running container subscribe to all the entities from which it should receive notifications

# define the POST_DATA_URL
POST_DATA_URL="http://<your_ip_address>:<PORT>/explanation/post_data"

echo "POST_DATA_URL: '$POST_DATA_URL'"

# PersonVehicleDetection subscription
curl -X POST 'https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/subscriptions' \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "urn:ngsi-ld:Parter_name:SubscriptionID",
    "type": "Subscription",
    "name": "Subscription name",
    "description": "Subscription description",
    "entities": [
        {
            "type": "PersonVehicleDetection"
        }
    ],
    "notification": {
        "endpoint": {
            "uri": "$POST_DATA_URL", 
            "accept": "application/json"
        }
    }
}'
