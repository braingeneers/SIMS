FROM pytorch/pytorch

WORKDIR /src

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y --no-install-recommends \
        curl \
        sudo \
        vim 

RUN curl -L https://bit.ly/glances | /bin/bash

RUN conda install --yes boto3 tenacity pandas numpy pip plotly scipy && \
    conda install -c conda-forge python-kaleido 

RUN conda update conda
RUN conda install python=3.8
RUN conda install anaconda-client
RUN conda update anaconda

RUN pip install matplotlib 
RUN pip install seaborn 
RUN pip install pytorch-lightning 
RUN pip install comet_ml 
RUN pip install wandb 
RUN pip install pytorch-tabnet
RUN pip install scanpy 
RUN pip install anndata
RUN pip install sklearn 

# Is this breaking everything??
# RUN pip install scvi-tools 

COPY . .