FROM nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        git \
        curl \
        wget \
        vim \
        gzip \
        python3.8 \
        python3.8-distutils \
        gcc-8 \
        g++-8 \
        gcc-10 \
        g++-10

RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python3.8 ./get-pip.py

RUN pip install numpy
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
RUN pip install easydict
RUN pip install tqdm
RUN pip install progress

RUN rm ./get-pip.py

ENV APP_HOME=/app

RUN mkdir $APP_HOME

WORKDIR $APP_HOME

ENTRYPOINT ["tail", "-f", "/dev/null"]
