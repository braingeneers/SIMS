FROM anibali/pytorch:1.10.2-cuda11.3
USER root

WORKDIR /src

RUN sudo apt-get update
RUN sudo apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get --allow-releaseinfo-change update && \
    sudo apt-get install -y --no-install-recommends \
    curl \
    sudo \
    vim

RUN curl -L https://bit.ly/glances | /bin/bash

RUN pip install matplotlib \
    seaborn \
    pytorch-lightning \
    comet_ml \
    wandb \
    pytorch-tabnet \
    scanpy \
    anndata \
    sklearn \
    boto3 \ 
    tenacity \ 
    pandas \
    plotly \
    scipy

RUN pip install git+https://github.com/jlehrer1/sims

# Is this breaking everything
# RUN pip install scvi-tools

COPY . .