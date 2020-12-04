#!/bin/sh
docker exec research_embedding python /workspace/app/django/manage.py runserver 0.0.0.0:8000
