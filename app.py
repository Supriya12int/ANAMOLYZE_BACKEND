from flask import Flask, request, jsonify
from flask_cors import CORS
import bcrypt
import json
import os
import gdown
from PIL import Image
import torch
import timm
from torchvision import transforms

app = Flask(__name__)
CORS(app)

USER_DB = "users.json"
MODEL_PATH = "swin_transformer_trained.pth"
GOOGLE_DRIVE_FILE_ID = "1mpmYQdydILZLM82MRHB5WCa3oc_E7mZK"

# Automatically download the model from Google Drive if not present
# if not os.path.exists(MODEL_PATH):
#     print("Model not found. Downloading from Google Drive...")
#     gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# User DB
if not os.path.exists(USER_DB):
    with open(USER_DB, "w") as f:
        json.dump([], f)

def read_users():
    with open(USER_DB, "r") as f:
        return json.load(f)

def write_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=4)

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    users = read_users()
    if any(user["username"] == username for user in users):
        return jsonify({"success": False, "message": "User already exists"}), 400

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users.append({"username": username, "password": hashed.decode('utf-8')})
    write_users(users)
    return jsonify({"success": True, "message": "User registered successfully"})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    users = read_users()
    user = next((u for u in users if u["username"] == username), None)
    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
        return jsonify({"success": True, "message": "Login successful"})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

# === Load Model ===
device = torch.device("cpu")
print("Creating model...")
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)

try:
    print("Loading model weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device,weights_only=False))
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:")
    print(e)

model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()

    class_names = ["defect", "normal"]
    predicted_label = class_names[pred_class]

    return jsonify({"prediction": predicted_label})


if __name__ == "__main__":
    app.run(debug=True)
