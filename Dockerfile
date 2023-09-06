FROM anibali/pytorch:1.13.0-cuda11.8
USER root

WORKDIR /src

RUN sudo apt-get update
RUN sudo apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get --allow-releaseinfo-change update && \
    sudo apt-get install -y --no-install-recommends \
    curl \
    sudo \
    vim

RUN pip install matplotlib \
    seaborn \
    pytorch-lightning \
    comet_ml \
    wandb \
    pytorch-tabnet \
    scanpy \
    anndata \
    scikit-learn \
    boto3 \ 
    tenacity \ 
    pandas \
    plotly \
    scipy

ENV cache=0
RUN pip3 install --use-pep517 --no-cache git+https://github.com/braingeneers/SIMS.git
COPY . .