#!/bin/bash
if [[ -z "$GITHUB_TOKEN" ]]; then
  echo "Error: GITHUB_TOKEN no está definido"
  exit 1
else
  echo "Configurando git con token de GitHub..."
  git config --global url."https://x-access-token:$GITHUB_TOKEN@github.com/end-to-end-mlops-databricks-3".insteadOf "https://github.com/end-to-end-mlops-databricks-3"
fi
