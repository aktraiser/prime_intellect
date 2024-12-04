#!/bin/bash

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Update pip
pip install --upgrade pip

# Clean existing installations
pip uninstall -y unsloth
pip uninstall -y transformers
pip uninstall -y peft
pip uninstall -y accelerate
pip uninstall -y bitsandbytes

# Install dependencies in the correct order
pip install transformers>=4.37.2
pip install accelerate
pip install bitsandbytes>=0.41.1
pip install "unsloth[cuda] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" "trl<0.9.0" peft
pip install evaluate scikit-learn scipy joblib threadpoolctl

# Execute the Python script
python training.py
