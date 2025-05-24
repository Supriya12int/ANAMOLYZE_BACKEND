#!/bin/bash

# Install model using gdown
echo "Downloading model..."
gdown --fuzzy --id 1mpmYQdydILZLM82MRHB5WCa3oc_E7mZK -O swin_transformer_trained.pth
ls -lh swin_transformer_trained.pth


# Start Flask app
echo "Starting Flask server..."
python app.py
