#!/usr/bin/env bash
# run_opencv_container.sh

IMAGE_NAME="opencv-gpu-build"
CONTAINER_NAME="opencv_cuda_dev"

if [ -z "$1" ]; then
  echo "Usage: $0 <host-directory-to-mount>"
  exit 1
fi

HOST_DIR="$(realpath "$1")"
CONTAINER_DIR="/workspace"

if [ ! -d "$HOST_DIR" ]; then
  echo "Error: Directory '$HOST_DIR' does not exist."
  exit 1
fi

# Run container as 'admin' user (UID 1000), mounting the host dir
docker run -it --rm \
  --gpus all \
  --device=/dev/video0 \
  --network=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --name "${CONTAINER_NAME}" \
  -v "${HOST_DIR}:${CONTAINER_DIR}" \
  -w "${CONTAINER_DIR}" \
  --user 1000:1000 \
  "${IMAGE_NAME}" \
  /bin/bash
