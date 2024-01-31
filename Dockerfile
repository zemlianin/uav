FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

RUN apt-get update  &&  DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && apt-get install -y \
    libopencv-dev \
    python3 \
    python3-pip \
    neovim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt

COPY inference_bench.py /app
COPY yolov8n.pt /app
COPY runs /app
COPY test_gen /app

WORKDIR /app

CMD ["tail", "-f", "/dev/null"]