from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# === Initialisation du pipeline DistilGPT2 ===
# Le téléchargement ne se fait qu'une seule fois, puis le modèle est caché dans ~/.cache/huggingface
generator = pipeline(
    "text-generation",
    model="distilbert/distilgpt2",
    device=-1,               # CPU seulement
    max_length=10,          # limite la longueur de sortie
    clean_up_tokenization_spaces=True,
)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Aucune question fournie"}), 400

    # Génération de texte
    output = generator(question, do_sample=False, num_return_sequences=1)
    # L'API renvoie une liste de dicts {"generated_text": "..."}
    text = output[0].get("generated_text", "")
    return jsonify({"response": text})

@app.route("/", methods=["GET"])
def index():
    return "DistilGPT2 API prête 🎉"

if __name__ == "__main__":
    # Utilise python3 dans le Procfile: web: python3 app.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    
