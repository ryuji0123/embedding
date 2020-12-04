#!/bin/sh
docker run -it --rm \
	--name research_embedding \
	-p 8000:8000 \
	-p 18888:18888 \
	-v ~/research-embedding:/workspace \
	-v $(readlink -f ./results/):$(readlink -f ./results/) \
	embedding bash
