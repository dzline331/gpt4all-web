from flask import Flask, request, jsonify
from gpt4all import GPT4All
import os
import requests

app = Flask(__name__)

# === Configuration du modèle ===
model_url = "https://huggingface.co/orel12/ggml-gpt4all-j-v1.3-groovy/resolve/main/ggml-gpt4all-j-v1.3-groovy.q4_0.bin"
model_dir = "models"
model_filename = "ggml-gpt4all-j-v1.3-groovy.q4_0.bin"
model_path = os.path.join(model_dir, model_filename)

# === Téléchargement automatique du modèle si nécessaire ===
if not os.path.exists(model_path):
    os.makedirs(model_dir, exist_ok=True)
    print("Téléchargement du modèle...")
    response = requests.get(model_url, stream=True)
    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Téléchargement terminé.")

# === Chargement du modèle GPT4All ===
model = GPT4All(model_path, model_name="gpt4all-custom", allow_download=False)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    response = model.generate(question)
    return jsonify({"response": response})

@app.route("/", methods=["GET"])
def index():
    return "GPT4All API fonctionne !"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
