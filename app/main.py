from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

model_path = "model/vansh_poet_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

@app.route('/generate', methods=['POST'])
def generate_poem():
    data = request.get_json()
    prompt = data.get("prompt", "")
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    poem = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"generated_poem": poem})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
