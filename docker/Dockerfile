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
RUN curl -sL https://deb.nodesource.com/setup_15.x | sudo -E bash -&& \
    apt-get install -y nodejs

# support plotly on jupyter lab
RUN jupyter labextension install jupyterlab-plotly 

# django
ENV DJANGO_PATH $WORK_PATH/app/server
COPY app/server/requirements.txt $DJANGO_PATH/
RUN pip install ${PIP_OPTIONS} -r $DJANGO_PATH/requirements.txt

# poincare-embedding
ENV POINCARE_PATH $WORKDIR/facebook_poincare_embeddings
WORKDIR $POINCARE_PATH
COPY facebook_poincare_embeddings $POINCARE_PATH
RUN python $POINCARE_PATH/setup.py build_ext --inplace

# duser setting
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID duser && \
		adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID duser
RUN adduser duser sudo && \
		echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER duser
WORKDIR $WORK_PATH
