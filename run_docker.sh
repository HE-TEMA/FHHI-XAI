#!/usr/bin/env bash
set -eu

# Override these without editing the script, for example:
# HOST_PORT=8081 CONTAINER_NAME=explanation_tfa02_8081 ./run_docker.sh
HOST_PORT="${HOST_PORT:-8080}"
CONTAINER_NAME="${CONTAINER_NAME:-explanation_tfa02}"
ENTITIES_TO_EXPLAIN="${ENTITIES_TO_EXPLAIN:-FloodSegmentation,PersonVehicleDetection}"

docker run --rm \
  -p "${HOST_PORT}:8080" \
  --name "${CONTAINER_NAME}" \
  -e "ENTITIES_TO_EXPLAIN=${ENTITIES_TO_EXPLAIN}" \
  --runtime=nvidia \
  --gpus all \
  explanation_tfa02
