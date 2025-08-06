# Multi-stage Docker build for HDC Robot Controller

# Build stage
FROM ubuntu:22.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    pkg-config \
    libgtest-dev \
    libeigen3-dev \
    libopencv-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y ros-humble-desktop \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy source code
WORKDIR /workspace
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Build C++ components
RUN mkdir -p build && cd build \
    && cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON .. \
    && ninja

# Install Python package
RUN pip install -e .

# Run tests during build
RUN cd build && ctest --output-on-failure
RUN python -m pytest test/ -v

# Runtime stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libopencv-dev \
    libeigen3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 runtime
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y ros-humble-ros-base python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash hdc_user && \
    usermod -aG sudo hdc_user

# Copy built artifacts from builder stage
COPY --from=builder /workspace/build/lib* /usr/local/lib/
COPY --from=builder /workspace/install /opt/hdc_robot_controller
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy source and examples
WORKDIR /home/hdc_user/hdc_robot_controller
COPY --chown=hdc_user:hdc_user examples/ examples/
COPY --chown=hdc_user:hdc_user launch/ launch/
COPY --chown=hdc_user:hdc_user config/ config/
COPY --chown=hdc_user:hdc_user README.md .

# Set up environment
ENV PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV ROS_DISTRO=humble

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
source /opt/ros/humble/setup.bash\n\
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH\n\
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

USER hdc_user
ENTRYPOINT ["/entrypoint.sh"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import hdc_robot_controller; hdc_robot_controller.validate_installation()" || exit 1

# Default command
CMD ["python3", "-c", "import hdc_robot_controller as hdc; print('HDC Robot Controller ready!'); hdc.get_info()"]

# Metadata
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>"
LABEL description="HDC Robot Controller - Hyperdimensional Computing for Robust Robotic Control"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/danieleschmidt/HDC-Robot-Controller"
LABEL org.opencontainers.image.documentation="https://hdc-robot-controller.readthedocs.io"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

# CUDA variant
FROM runtime as cuda-runtime
USER root

# Install CUDA runtime
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-0 \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.0-1_all.deb

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Rebuild with CUDA support
WORKDIR /tmp/build
COPY --from=builder /workspace .
RUN WITH_CUDA=1 pip install -e .[cuda]

USER hdc_user
WORKDIR /home/hdc_user/hdc_robot_controller