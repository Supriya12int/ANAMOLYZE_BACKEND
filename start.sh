#!/bin/bash

# Install model using gdown
echo "Downloading model..."
gdown --fuzzy "$MODEL_URL" -O swin_transformer_trained.pth
ls -lh swin_transformer_trained.pth


# Start Flask app
echo "Starting Flask server..."
python app.py
