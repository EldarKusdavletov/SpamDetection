# SpamDetection

SpamDetection is a TensorFlow-based text classification project for **Russian spam risk detection**.  
It trains an LSTM model on a labeled dataset and provides two inference interfaces:

- a **CLI predictor** (`predict.py`) for single-message checks;
- a **Flask HTTP API** (`server.py`) for integration into external systems.

The repository already includes trained artifacts (`spam_classifier_tensorflow.keras` and `tokenizer_tensorflow.pickle`) so you can run predictions immediately.

---

## What the project does

Given an input message, the model returns a **risk score** in the range `[0, 1]`:

- `0.0` → very low spam risk;
- `1.0` → very high spam risk.

In CLI mode, this score is also mapped to a human-readable label:

- `<= 0.3`: `safe` (`✅ Безопасно`)
- `0.3 - 0.7`: `suspicious` (`⚠️ Сомнительно`)
- `> 0.7`: `high_risk` (`🚨 Высокий риск`)

---

## Repository structure

- `/home/runner/work/SpamDetection/SpamDetection/train.py`  
  Trains and evaluates the model, then saves model/tokenizer artifacts.
- `/home/runner/work/SpamDetection/SpamDetection/predict.py`  
  Loads artifacts and predicts risk for a message from command-line arguments.
- `/home/runner/work/SpamDetection/SpamDetection/server.py`  
  Runs a Flask API with `/analyze` and `/health` endpoints.
- `/home/runner/work/SpamDetection/SpamDetection/requirements.txt`  
  Pinned dependencies (TensorFlow, scikit-learn, pandas, Flask, etc.).
- `/home/runner/work/SpamDetection/SpamDetection/spam_classifier_tensorflow.keras`  
  Trained TensorFlow model file.
- `/home/runner/work/SpamDetection/SpamDetection/tokenizer_tensorflow.pickle`  
  Serialized tokenizer used during training.

---

## Model and data pipeline

### Data source

Training data is loaded from:

`hf://datasets/darkQibit/russian-spam-detection/processed_combined.parquet`

Expected dataset columns:

- `message` — input text
- `label` — binary target

### Text preprocessing

`train.py` uses Keras `Tokenizer` and sequence padding:

- vocabulary size: `5000`
- max sequence length: `150`
- OOV token: `<unk>`
- post-padding and post-truncation

### Model architecture

Sequential network:

1. `Embedding(input_dim=5000, output_dim=128)`
2. `LSTM(64, dropout=0.2, recurrent_dropout=0.2)`
3. `Dropout(0.5)`
4. `Dense(1, activation='sigmoid')`

Compilation:

- optimizer: `adam`
- loss: `binary_crossentropy`
- metric: `accuracy`

Training settings:

- train/test split: `80/20`
- batch size: `32`
- epochs: `10`
- early stopping on validation loss (`patience=3`, restore best weights)

---

## Setup

### 1) Create and activate virtual environment

```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Option A: Use pre-trained artifacts (fast start)

Run:

```bash
python predict.py "ваше сообщение"
```

### Option B: Retrain model

Run:

```bash
python train.py
```

This regenerates:

- `spam_classifier_tensorflow.keras`
- `tokenizer_tensorflow.pickle`

---

## CLI examples

```bash
python3 predict.py "я устал брать деньги в долги и кредиты"
```

Expected-style output:

```text
--- Результат анализа ---
Сообщение:      'я устал брать деньги в долги и кредиты'
Оценка риска:   0.1497 (14.97%)
Уровень угрозы: ✅ Безопасно
-------------------------
```

```bash
python3 predict.py "хочешь взять долги и кредиты без процентов"
```

```text
--- Результат анализа ---
Сообщение:      'хочешь взять долги и кредиты без процентов'
Оценка риска:   0.5812 (58.12%)
Уровень угрозы: ⚠️ Сомнительно (требуется проверка)
-------------------------
```

---

## API server

Start server:

```bash
python3 server.py
```

Server listens on `0.0.0.0:7503`.

### Health check

```bash
curl http://127.0.0.1:7503/health
```

Response:

```json
{"status":"ok"}
```

### Analyze message

```bash
curl -X POST http://127.0.0.1:7503/analyze \
  -H "Content-Type: application/json" \
  -d '{"message":"хочешь взять долги и кредиты без процентов"}'
```

Response:

```json
{"risk_score":0.5811898708343506}
```

### API validation behavior

- non-JSON request → `400`
- missing `message` field → `400`
- non-string `message` → `400`
- prediction errors → `500`

---

## Notes and limitations

- The classifier is specialized for **Russian-language spam patterns**; quality may drop on other languages/domains.
- The model outputs a probability-like score, not a legal/fraud certainty.
- Thresholds in `predict.py` are heuristic and can be tuned for your tolerance to false positives/negatives.
- `server.py` loads artifacts at startup; ensure model and tokenizer files exist before launching.

---

## Recommended next improvements

- Add reproducible experiment tracking (metrics history, dataset version, seed control).
- Add automated tests for API input validation and inference contract.
- Add model evaluation metrics beyond accuracy (precision/recall/F1, ROC-AUC, confusion matrix).
- Add Docker packaging for deployment.