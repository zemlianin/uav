FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN apt-get update  &&  DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && apt-get install -y \
    libopencv-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt

COPY . /app

WORKDIR /app

CMD ["tail", "-f", "/dev/null"]
