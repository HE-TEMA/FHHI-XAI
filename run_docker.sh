# docker run -p <port_exposed_on_cloud_machine>:<PORT> \
docker run -p 8080:8080 \
--name explanation_tfa02 \
--runtime=nvidia \
--gpus all \
explanation_tfa02
