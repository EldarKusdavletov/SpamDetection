import pickle
import json
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Константы ---
MAX_LEN = 150
MODEL_PATH = 'spam_classifier_tensorflow.keras'
TOKENIZER_PATH = 'tokenizer_tensorflow.pickle'

app = Flask(__name__)

# --- Загрузка артефактов ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Модель и токенизатор успешно загружены")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    print("Убедитесь, что файлы модели и токенизатора существуют")


# --- Логика анализа текста ---
def analyze_text(text: str):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    risk_score = model.predict(padded_sequence, verbose=0)[0][0]

    return {"risk_score": float(risk_score)}


@app.route('/analyze', methods=['POST'])
def analyze():
    if not request.is_json:
        return jsonify({"error": "Запрос должен быть в формате JSON"}), 400

    data = request.get_json()

    if 'message' not in data:
        return jsonify({"error": "Поле 'message' обязательно"}), 400

    message = data['message']

    if not isinstance(message, str):
        return jsonify({"error": "Поле 'message' должно быть строкой"}), 400

    try:
        result = analyze_text(message)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Ошибка при анализе: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7503, debug=False)
