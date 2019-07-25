FROM nvidia/cuda:9.0-cudnn7-devel

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends  \
    gcc \
    git \
    make \
    wget \
    curl \
    unzip \
    cmake \
    libsm6 \
    libxext6 \
    libpq-dev \
    libxrender1 \
    libglib2.0-0 \
    liblapack-dev \
    libxrender-dev \
    libopenblas-dev \
    build-essential \
    ca-certificates \
    python-pip \
    python2.7 \
    python2.7-dev \
    python-tk \
    && curl -kL https://bootstrap.pypa.io/get-pip.py | python

# pip
COPY ./requirements.txt /tmp
RUN pip install --upgrade pip setuptools \
    && pip install -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements.txt

# ffmpeg
RUN mkdir /tmp/ffmepg \
    && wget -O /tmp/ffmepg/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    && tar xvf /tmp/ffmepg/ffmpeg.tar.xz -C /tmp/ffmepg --strip-components 1 \
    && cp /tmp/ffmepg/ffmpeg /usr/local/bin \
    && cp /tmp/ffmepg/ffprobe /usr/local/bin \
    && rm -rf /tmp/ffmepg

RUN apt-get clean \
    && rm -rf /var/chache/apt/archives/* /var/lib/apt/lists/*


