
import string
import re
import joblib
from flask import Flask, request, jsonify
def remove_pun(text):
    if isinstance(text, str):
        for pun in string.punctuation:
            text = text.replace(pun, " ")
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
    return text
app = Flask(__name__)
model_pipe = None
def load_model():
    global model_pipe
    try:
        model_filename = 'skill_model_pipeline.joblib'
        model_pipe = joblib.load(model_filename)
        print(f"SUCCESS: Model '{model_filename}' loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Model file '{model_filename}' not found. API cannot make predictions.")
    except Exception as e:
        print(f"ERROR: Could not load model '{model_filename}'. Exception: {e}")
load_model()
@app.route('/')
def home():
    return "Welcome to the Skill Prediction API! Use the /predict endpoint (POST) to get predictions."
@app.route('/predict', methods=['POST'])
def predict_skill():
    if model_pipe is None:
        return jsonify({"error": "Model not loaded. Server is not ready to make predictions."}), 503
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format"}), 400
    data = request.get_json()
    if 'skills_text' not in data:
        return jsonify({"error": "Missing 'skills_text' field in JSON payload"}), 400
    input_text = data['skills_text']
    if not isinstance(input_text, str) or not input_text.strip():
        return jsonify({"error": "'skills_text' must be a non-empty string"}), 400
    try:
        processed_text = remove_pun(input_text)
        prediction_result = model_pipe.predict([processed_text])
        predicted_label = prediction_result[0]
        if hasattr(predicted_label, 'item'):
            predicted_label = predicted_label.item()
        return jsonify({
            "input_text": input_text,
            "processed_text": processed_text,
            "predicted_category_id": predicted_label
        }), 200
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "An internal server error occurred during prediction."}), 500
if __name__ == '__main__':
    print("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
