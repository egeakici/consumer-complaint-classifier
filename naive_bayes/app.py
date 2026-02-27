import os
import pickle
import time

from flask import Flask, request, jsonify
from flasgger import Swagger

from preprocess_for_inference import preprocess
import config

app = Flask(__name__)

swagger_config = {
    "headers": [],
    "specs": [{"endpoint": "apispec", "route": "/apispec.json"}],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs",
}
swagger_template = {
    "info": {
        "title":       "Consumer Complaint Classifier — Naive Bayes",
        "description": "Pipeline: Raw text → Preprocess (Stemming) → TF-IDF → Predict → Yes / No",
        "version":     "1.0.0",
    },
    "consumes": ["application/json"],
    "produces": ["application/json"],
}
swagger = Swagger(app, config=swagger_config, template=swagger_template)

print("[startup] Loading model...")
with open(config.MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(config.VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
print(f"[startup] Ready. → http://localhost:{config.FLASK_PORT}/apidocs")


def _predict_single(raw_text: str) -> dict:
    t0           = time.perf_counter()
    cleaned_text = preprocess(raw_text)
    tfidf_vec    = vectorizer.transform([cleaned_text])
    prediction   = model.predict(tfidf_vec)[0]
    return {
        "prediction":  prediction,
        "text_length": len(raw_text),
        "clean_text":  cleaned_text,
        "latency_ms":  round((time.perf_counter() - t0) * 1000, 2),
    }


@app.route("/health", methods=["GET"])
def health():
    """
    Check server and model status.
    ---
    tags:
      - System
    responses:
      200:
        description: Server is running
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            model:
              type: string
              example: ComplementNB
            classes:
              type: array
              items:
                type: string
              example: ["No", "Yes"]
    """
    return jsonify({
        "status":  "ok",
        "model":   type(model).__name__,
        "classes": list(model.classes_),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether a single complaint will be disputed.
    ---
    tags:
      - Prediction
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - text
          properties:
            text:
              type: string
              example: "I contacted the bank multiple times but received no response."
    responses:
      200:
        description: Prediction successful
        schema:
          type: object
          properties:
            prediction:
              type: string
              example: "Yes"
            text_length:
              type: integer
              example: 62
            clean_text:
              type: string
              example: "contact bank multipl time receiv respons"
            latency_ms:
              type: number
              example: 3.14
      400:
        description: Invalid request
      500:
        description: Server error
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json."}), 400
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Invalid JSON body."}), 400
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "'text' field is required and cannot be empty."}), 400
    try:
        return jsonify(_predict_single(text)), 200
    except Exception as exc:
        return jsonify({"error": f"Prediction error: {str(exc)}"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict dispute likelihood for multiple complaints at once.
    ---
    tags:
      - Prediction
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - texts
          properties:
            texts:
              type: array
              items:
                type: string
              example:
                - "The company ignored my complaint completely."
                - "Issue was resolved promptly, no further action needed."
    responses:
      200:
        description: Batch prediction successful
        schema:
          type: object
          properties:
            predictions:
              type: array
              items:
                type: object
            count:
              type: integer
              example: 2
      400:
        description: Invalid request
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json."}), 400
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Invalid JSON body."}), 400
    texts = body.get("texts")
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a non-empty list."}), 400
    if len(texts) > 100:
        return jsonify({"error": "Maximum 100 texts per request."}), 400
    try:
        results = [_predict_single(str(t)) for t in texts]
        return jsonify({"predictions": results, "count": len(results)}), 200
    except Exception as exc:
        return jsonify({"error": f"Prediction error: {str(exc)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host=config.FLASK_HOST, port=config.FLASK_PORT)
