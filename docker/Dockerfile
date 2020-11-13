FROM pytorch/pytorch
# apt-get
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq && \
		apt-get install -y \
			build-essential curl lsof git sudo tmux tree vim wget && \
		apt-get clean && \
		rm -rf /var/lib/apt/lists/* && \
		rm -rf /var/cache/apk/*

# python
ENV WORK_PATH /workspace
WORKDIR $WORK_PATH
ENV PYTHONPATH "/workspace:${PYTHONPATH}"
ENV PIP_OPTIONS "--no-cache-dir --progress-bar off"
COPY docker/requirements.txt $WORK_PATH/docker/
RUN pip install ${PIP_OPTIONS} -r $WORK_PATH/docker/requirements.txt && \
		pip install ${PIP_OPTIONS} -U setuptools && \
		jupyter serverextension enable --py jupyterlab

# install nodejs for plotly
RUN curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash - \
    && sudo apt-get install -y nodejs

# support plotly on jupyter lab
RUN jupyter labextension install jupyterlab-plotly 

# django
COPY app/requirements.txt $WORK_PATH/app/
RUN pip install ${PIP_OPTIONS} -r $WORK_PATH/app/requirements.txt

# poincare-embedding
COPY 
