from flask import Flask, request, jsonify
from gpt4all import GPT4All

app = Flask(__name__)
model = GPT4All("models/ggml-gpt4all-j-v1.3-groovy.bin")  # modèle local

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    response = model.generate(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
