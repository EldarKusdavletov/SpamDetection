import sys
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Константы ---
MAX_LEN = 150
MODEL_PATH = 'spam_classifier_tensorflow.keras'
TOKENIZER_PATH = 'tokenizer_tensorflow.pickle'


# --- 1. Загрузка артефактов ---
def load_artifacts():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except (IOError, FileNotFoundError):
        print(f"🚨 Ошибка: Не найдены файлы {MODEL_PATH} или {TOKENIZER_PATH}.")
        print("💡 Сначала запустите train_tensorflow.py для обучения модели.")
        sys.exit(1)


# --- 2. Логика анализа текста ---
def analyze_text(text: str, model, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    risk_score = model.predict(padded_sequence, verbose=0)[0][0]

    if risk_score <= 0.3:
        risk_level = "safe"
        risk_description = "✅ Безопасно"
    elif 0.3 < risk_score <= 0.7:
        risk_level = "suspicious"
        risk_description = "⚠️ Сомнительно (требуется проверка)"
    else:
        risk_level = "high_risk"
        risk_description = "🚨 Высокий риск (вероятный мошенник)"

    return {
        "input_text": text,
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "risk_description": risk_description
    }


# --- Основной блок ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python analyze_cli_tensorflow.py \"ваше сообщение\"")
        sys.exit(1)

    message = " ".join(sys.argv[1:])

    model, tokenizer = load_artifacts()
    result = analyze_text(message, model, tokenizer)

    print("\n--- Результат анализа ---")
    print(f"Сообщение:      '{result['input_text']}'")
    print(f"Оценка риска:   {result['risk_score']:.4f} ({result['risk_score'] * 100:.2f}%)")
    print(f"Уровень угрозы: {result['risk_description']}")
    print("-------------------------")