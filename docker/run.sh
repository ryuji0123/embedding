#!/bin/sh
docker stop research_embedding
docker run -dit --rm \
	--name research_embedding \
	--gpus all \
	-p 3000:3000 \
	-p 8000:8000 \
	-p 18885:18885 \
	-v $PWD:/workspace \
	-v $(readlink -f ./results/):$(readlink -f ./results/) \
	e94e24e16a03 bash
