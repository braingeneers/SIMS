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

RUN conda install --yes boto3 tenacity pandas numpy pip plotly scipy && \
    conda install -c conda-forge python-kaleido 

RUN pip install matplotlib 
RUN pip install seaborn 
RUN pip install pytorch-lightning 
RUN pip install comet_ml 
RUN pip install wandb 
RUN pip install pytorch-tabnet
RUN pip install scanpy 
RUN pip install anndata
RUN pip install sklearn 

# Is this breaking everything
# RUN pip install scvi-tools 

COPY . .