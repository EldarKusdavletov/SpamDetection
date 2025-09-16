import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Константы ---
VOCAB_SIZE = 5000
MAX_LEN = 150
EMBEDDING_DIM = 128
LSTM_UNITS = 64
NUM_EPOCHS = 10
BATCH_SIZE = 32
MODEL_PATH = 'spam_classifier_tensorflow.keras'
TOKENIZER_PATH = 'tokenizer_tensorflow.pickle'
DATASET_PATH = "hf://datasets/darkQibit/russian-spam-detection/processed_combined.parquet"


# --- 1. Загрузка и подготовка данных ---

def load_data(path: str) -> pd.DataFrame:
    try:
        print("Загрузка датасета...")
        return pd.read_parquet(path)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return pd.DataFrame()


def preprocess_data(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    return train_padded, test_padded, y_train.values, y_test.values, tokenizer


# --- 2. Архитектура и обучение модели ---

def build_model():
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- Основной скрипт ---
if __name__ == "__main__":
    df = load_data(DATASET_PATH)

    X_train, X_test, y_train, y_test, tokenizer = preprocess_data(df)

    print("Создание и компиляция модели...")
    model = build_model()
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("Начало обучения...")
    history = model.fit(
        X_train, y_train,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    print("\nОценка модели на тестовых данных:")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Потери: {loss:.4f} | Точность: {accuracy * 100:.2f}%")

    # Сохранение модели и токенизатора
    model.save(MODEL_PATH)
    print(f"\n💾 Модель сохранена в {MODEL_PATH}")

    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"💾 Токенизатор сохранен в {TOKENIZER_PATH}")
