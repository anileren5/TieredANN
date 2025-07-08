FROM ubuntu:jammy

# Install dependencies
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:git-core/ppa && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
        git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev \
        clang-format libboost-dev libboost-program-options-dev \
        libmkl-full-dev libcpprest-dev python3.10

# Create /app directory and copy all contents there
WORKDIR /app
COPY . .

# Set working directory for following commands
WORKDIR /app