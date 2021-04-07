FROM continuumio/anaconda3:2019.10
LABEL author=DanielJunior email="danieljunior@id.uff.br"
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get -y install gcc g++ nano cython xvfb build-essential \
    libgtk-3-dev wget libdbus-glib-1-2 libprocps-dev procps unzip libblas-dev liblapack-dev

ENV LANG pt-BR.UTF-8
ENV LANGUAGE pt-BR.UTF-8

ENV PATH /opt/conda/envs/env/bin:$PATH
ENV PATH /opt/conda/bin:$PATH

RUN conda update conda \ 
    && conda create -y -n env python=3.7 \
    && echo "source activate base" > ~/.bashrc
RUN /bin/bash -c "source activate base"

RUN mkdir -p /app
COPY . /app
WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --ignore-installed --no-cache-dir -r requirements.txt 

RUN gdown https://drive.google.com/uc?id=1s4T8Y9QUNw-iKTjYfGpLhIJMAL18OWV5 && \
    unzip models.zip && \
    rm -f models.zip

RUN gdown https://drive.google.com/uc?id=1B-k_yGvUxkCjA009XWu9f2Untm_YN-o_ && \
    unzip datasets.zip && \
    rm -f datasets.zip

RUN conda install -c conda-forge black nodejs=12
RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager && \ 
    jupyter labextension install jupyterlab-python-file --no-build && \ 
    jupyter nbextension enable --py widgetsnbextension --sys-prefix && \ 
    conda clean --all -f -y && \ 
    jupyter lab build && \ 
    jupyter lab clean && \ 
    npm cache clean --force
RUN rm -rf /var/lib/apt/lists/* && apt-get clean

EXPOSE 8888
