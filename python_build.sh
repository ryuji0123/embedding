#!/bin/sh
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude app/,notebooks/,facebook_poincare_embeddings/
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude app/,notebooks/,facebook_poincare_embeddings/
pytest --ff --log-cli-level=10 --durations=0 --ignore app/,notebooks/,facebook_poincare_embeddings/
