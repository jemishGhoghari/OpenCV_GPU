# OpenCV with CUDA in Docker

[![Ubuntu 22.04](https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu)](https://releases.ubuntu.com/22.04/)
[![CUDA 12.2](https://img.shields.io/badge/CUDA-12.2-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![cuDNN 8](https://img.shields.io/badge/cuDNN-8-76B900?logo=nvidia)](https://developer.nvidia.com/cudnn)
[![OpenCV 4.12.0](https://img.shields.io/badge/OpenCV-4.12.0-5C3EE8?logo=opencv)](https://github.com/opencv/opencv)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://www.docker.com/)

A Docker environment for building and running OpenCV 4.12.0 with CUDA/cuDNN GPU acceleration on Ubuntu 22.04.

---

## What's Included

- OpenCV 4.12.0 built with `opencv_contrib` modules
- CUDA 12.2 + cuDNN 8 with DNN GPU backend enabled
- Hardware acceleration: OpenGL, GTK, GStreamer, FFmpeg, V4L2, dc1394, OpenEXR
- Dev tools: `cmake`, `gcc/g++`, `clang`, `ninja`, `git`, `gdb`, `ccache`, `python3`
- Non-root `admin` user with passwordless sudo

---

## Requirements

- **Host OS:** Linux (tested on Ubuntu 22.04 / 24.04)
- **NVIDIA GPU driver:** version **≥ 535**
- **Docker Engine** + **NVIDIA Container Toolkit**
  → [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Check your driver version:
```bash
nvidia-smi
```

---

## Getting Started

### 1. Build the Docker image

```bash
docker build -t opencv-gpu-build .
```

This will clone and compile OpenCV from source inside the image. It takes a while.

### 2. Run the container

Use the provided script and pass a local directory to mount as your workspace:

```bash
./run_dev.sh /path/to/your/project
```

This mounts your directory at `/workspace` inside the container and drops you into a bash shell.

> **Note:** The script also forwards your display (`$DISPLAY`) and `/dev/video0` for GUI windows and webcam access. Make sure X11 forwarding is allowed on the host if needed:
> ```bash
> xhost +local:docker
> ```

### 3. Verify GPU access

Once inside the container, run:

```bash
python3 -c "import cv2; print('OpenCV:', cv2.__version__); print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

You should see your OpenCV version and a non-zero CUDA device count.

---

## GPU Architecture

The Dockerfile defaults to CUDA architectures `86;89;90` (RTX 30/40 series). If your GPU is different, set the build arg:

```bash
docker build --build-arg CMAKE_CUDA_ARCHITECTURES=75 -t opencv-gpu-build .
```

Common values: `75` (RTX 20xx), `86` (RTX 30xx), `89` (RTX 40xx).

---

## Building the Example Code

The `opencv_example/` directory contains a YOLO11 object detection demo using OpenCV's DNN module.

Inside the container:

```bash
cd /workspace/opencv_example
mkdir build && cd build
cmake ..
make -j$(nproc)
./main
```

The example uses `models/yolo11m.onnx` (included) and the COCO class list from `coco/coco.names`.

---

## Project Structure

```
.
├── Dockerfile                    # Builds the OpenCV + CUDA image
├── run_dev.sh                    # Helper script to run the container
├── opencv_example/               # YOLO11 object detection demo
│   ├── main.cpp
│   ├── CMakeLists.txt
│   ├── coco/                     # COCO class names and config
│   ├── models/                   # ONNX model files (yolo11m.onnx)
│   └── utils/                    # Helper scripts (pt_to_onnx, yaml_to_names)
└── opencv_lessons_code/          # Standalone lesson examples
    ├── lesson_1.cpp
    ├── lesson_2.cpp
    ├── histogram_calculation.cpp
    ├── shi-tomasi.cpp
    └── CMakeLists.txt
```

---

## Lesson Code

The `opencv_lessons_code/` directory contains standalone examples covering basic OpenCV GPU operations. Build them the same way as the main example:

```bash
cd /workspace/opencv_lessons_code
mkdir build && cd build
cmake ..
make -j$(nproc)
```
