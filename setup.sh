#!/bin/bash
# Create the private Python environment for the Odysseus image tools.
# Separate from Odysseus's own .venv so nothing there is touched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python \
  torch diffusers transformers accelerate safetensors huggingface_hub \
  pillow numpy opencv-python-headless "mcp<2"
.venv/bin/python -c "import torch,diffusers,cv2,mcp;print('torch',torch.__version__,'mps',torch.backends.mps.is_available(),'diffusers',diffusers.__version__)"
