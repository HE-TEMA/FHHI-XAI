# docker run -p <local_port>:<PORT> \
docker run -p 14:15 \
--name explanation_tfa02 \
-e BASE_PATH=/explanation \
-e TESTING='True' \
-e PORT=15 \
-e BROKER_URL=https://orion.tema.digital-enabler.eng.it \
-e CALLBACK_URL=/get_data \
explanation_tfa02