#!/bin/sh
docker run -it --rm -p 8000:8000 -p 18888:18888 -v ~/research-embedding:/workspace -v $(readlink -f ./results/):$(readlink -f ./results/) --name research_embedding embedding bash
