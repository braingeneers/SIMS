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

RUN pip install matplotlib 
RUN pip install pynndescent 
RUN pip install seaborn 
RUN pip install imbalanced-learn 
RUN pip install pytorch-lightning 
RUN pip install comet_ml 
RUN pip install wandb 
RUN pip install pytorch-tabnet
RUN pip install scanpy 
RUN pip install anndata
RUN pip install xgboost
RUN pip install scvi-tools 
RUN pip install sklearn 

COPY . .