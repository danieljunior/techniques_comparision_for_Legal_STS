FROM continuumio/anaconda3:2019.10
LABEL author=DanielJunior email="danieljunior@id.uff.br"

RUN apt-get --allow-releaseinfo-change update && \
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

ARG FIREFOX_VERSION=58.0.2
RUN wget --no-verbose --no-check-certificate -O /tmp/firefox.tar.bz2 https://download-installer.cdn.mozilla.net/pub/firefox/releases/$FIREFOX_VERSION/linux-x86_64/en-US/firefox-$FIREFOX_VERSION.tar.bz2 \
   && rm -rf /opt/firefox \
   && tar -C /opt -xjf /tmp/firefox.tar.bz2 \
   && rm /tmp/firefox.tar.bz2 \
   && mv /opt/firefox /opt/firefox-$FIREFOX_VERSION \
   && ln -fs /opt/firefox-$FIREFOX_VERSION/firefox /usr/bin/firefox
ARG GK_VERSION=v0.19.1
RUN wget --no-verbose --no-check-certificate -O /tmp/geckodriver.tar.gz http://github.com/mozilla/geckodriver/releases/download/$GK_VERSION/geckodriver-$GK_VERSION-linux64.tar.gz \
   && rm -rf /opt/geckodriver \
   && tar -C /opt -zxf /tmp/geckodriver.tar.gz \
   && rm /tmp/geckodriver.tar.gz \
   && mv /opt/geckodriver /opt/geckodriver-$GK_VERSION \
   && chmod 755 /opt/geckodriver-$GK_VERSION \
   && ln -fs /opt/geckodriver-$GK_VERSION /usr/bin/geckodriver

RUN mkdir -p /app
COPY . /app
RUN mkdir -p /app/results/tcu && mkdir -p /app/results/stj

WORKDIR /app

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL True
RUN pip install --upgrade pip && \
    pip install --ignore-installed --no-cache-dir -r requirements.txt 

RUN conda install -c conda-forge black nodejs=12 && npm config rm proxy && npm config rm https-proxy
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
#    jupyter labextension install jupyterlab-python-file --no-build && \
#    jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
#    conda clean --all -f -y && \
#    jupyter lab build --dev-build=False --minimize=False && \
#    jupyter lab clean && \
#    npm cache clean --force
RUN rm -rf /var/lib/apt/lists/* && apt-get clean

EXPOSE 8888
