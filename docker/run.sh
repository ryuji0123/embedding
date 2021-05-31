#!/bin/sh
docker run -it --rm \
	--name research_embedding \
	-p 3000:3000 \
	-p 8000:8000 \
	-p 18885:18885 \
	-v $PWD:/workspace \
	-v $(readlink -f ./results/):$(readlink -f ./results/) \
	embedding bash
