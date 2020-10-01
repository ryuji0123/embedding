# docker build -t ryookuda/embedding .
# docker run -it --rm -v ~/embedding:/workspace ryookuda/embedding bash

FROM pytorch/pytorch
# apt-get
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq && \
		apt-get install -y \
			curl lsof git tmux tree vim wget && \
		apt-get clean && \
		rm -rf /var/lib/apt/lists/* && \
		rm -rf /var/cache/apk/*

# python
ENV WORK_PATH /workspace
WORKDIR $WORK_PATH
ENV PYTHONPATH "/workspace:${PYTHONPATH}"
ENV PIP_OPTIONS "--no-cache-dir --progress-bar off"
COPY requirements.txt /workspace
RUN pip install ${PIP_OPTIONS} -r /workspace/requirements.txt && \
		pip install ${PIP_OPTIONS} -U setuptools
