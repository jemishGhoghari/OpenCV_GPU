# ==========================================================
# All-in-one Dev Image — CUDA 12.2 + OpenCV 4.12.0 (Ubuntu 22.04)
# ==========================================================
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.12.0
ARG CMAKE_CUDA_ARCHITECTURES="86;89;90"   # adjust/remove if unsure

# Base dev tools, Python, editors
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config \
    git curl wget unzip zip \
    gdb clang ccache \
    sudo vim nano \
    python3 python3-dev python3-pip python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Multimedia, vision, GUI, OpenGL/Qt, compression (dev + runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk-3-dev libeigen3-dev libtbb-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libdc1394-dev libopenexr-dev libv4l-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgl1-mesa-dev libglu1-mesa-dev \
    qtbase5-dev \
    zlib1g-dev libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

# cuDNN (runtime + headers) for CUDA 12.x
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Fetch OpenCV sources
WORKDIR /opt
RUN git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git && \
    git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git

# Configure & build OpenCV
WORKDIR /opt/opencv/build
RUN cmake -S .. -B . \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME=opencv4.pc \
    -D WITH_TBB=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_V4L=ON \
    -D WITH_GTK=ON \
    -D WITH_QT=ON \
    -D WITH_OPENGL=ON \
    -D WITH_1394=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_opencv_dnn=ON \
    -D CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
 && cmake --build . --config Release -j"$(nproc)" \
 && cmake --install . --prefix /usr/local

# Refresh linker cache AS ROOT before switching user
RUN ldconfig

# Create passwordless-sudo dev user
RUN useradd -m -u 1000 -s /bin/bash admin \
 && echo "admin ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/admin \
 && chmod 0440 /etc/sudoers.d/admin

USER admin
WORKDIR /home/admin

# Quick check on start (optional)
CMD ["/bin/bash","-lc","python3 - << 'PY'\nimport cv2\nprint('OpenCV:', cv2.__version__)\nprint('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())\nprint('CUDA enabled:', cv2.cuda.getCudaEnabledDeviceCount()>0)\nPY"]