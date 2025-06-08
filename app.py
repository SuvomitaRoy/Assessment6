from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification

app = Flask(__name__)

# Load your fine-tuned model
model_name = "distilgpt2"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
text_generator = pipeline("text-generation", model="your-model-name")

@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    response = text_generator(prompt, max_length=100)
    generated_text = response[0]["generated_text"]

    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    app.run(debug=True)
