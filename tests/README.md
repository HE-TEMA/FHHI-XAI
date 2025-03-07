# Testing receiving notifications and loading data

Endpoint `/explanation/post_data` will receive notifications from the Orion Context Broker (set in `../subscribe.sh`).
Here we test this endpoint by sending it some test entities.

As first step, download the test entities:
```bash
./download_auth_entities.sh
```

then run tests for each Entity
```bash
python test_post_data.py BurntSegmentation
python test_post_data.py FireSegmentation
python test_post_data.py FloodSegmentation
python test_post_data.py PersonVehicleDetection
python test_post_data.py SmokeSegmentation
```