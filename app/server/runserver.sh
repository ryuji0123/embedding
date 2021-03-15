#!/bin/sh
docker exec research_embedding python /workspace/app/server/manage.py runserver 0.0.0.0:8000
