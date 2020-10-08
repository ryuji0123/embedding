#!/bin/sh
docker run -it --rm -p 18888:18888 -v ~/research-embedding:/workspace --name research_embedding embedding bash
