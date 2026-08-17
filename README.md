# SpamDetection

SpamDetection is a production-oriented TensorFlow project for **Russian spam-risk detection** in text messages.
It combines deep-learning training with practical inference interfaces, so you can go from model idea to usable anti-spam signal fast.

It includes:
- **Model training** (`train.py`)
- **CLI inference** (`predict.py`)
- **HTTP API inference** (`server.py`)

Pretrained artifacts are already in the repository, so you can run real predictions right away.

## Quick start

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py "ваше сообщение"
```

## What the model returns

The model outputs a `risk_score` in the range `[0, 1]`:
- `0` = low spam risk
- `1` = high spam risk

In CLI mode (`predict.py`), this score is mapped to:
- `<= 0.3` → `safe` (`✅ Безопасно`)
- `0.3 < score <= 0.7` → `suspicious` (`⚠️ Сомнительно`)
- `> 0.7` → `high_risk` (`🚨 Высокий риск`)

## Project structure

- `/home/runner/work/SpamDetection/SpamDetection/train.py` — train/evaluate model and save artifacts
- `/home/runner/work/SpamDetection/SpamDetection/predict.py` — run single-message prediction from CLI
- `/home/runner/work/SpamDetection/SpamDetection/server.py` — Flask API (`/analyze`, `/health`)
- `/home/runner/work/SpamDetection/SpamDetection/requirements.txt` — Python dependencies
- `/home/runner/work/SpamDetection/SpamDetection/spam_classifier_tensorflow.keras` — trained model
- `/home/runner/work/SpamDetection/SpamDetection/tokenizer_tensorflow.pickle` — trained tokenizer

## Training pipeline

### Dataset
`train.py` loads:

`hf://datasets/darkQibit/russian-spam-detection/processed_combined.parquet`

Expected columns:
- `message` (text)
- `label` (binary target)

### Preprocessing
- Keras `Tokenizer(num_words=5000, oov_token="<unk>")`
- Sequence padding/truncation to `maxlen=150` (post mode)

### Model
Architecture:
1. `Embedding(5000, 128)`
2. `LSTM(64, dropout=0.2, recurrent_dropout=0.2)`
3. `Dropout(0.5)`
4. `Dense(1, activation="sigmoid")`

Training setup:
- `train_test_split(test_size=0.2, random_state=42)`
- `batch_size=32`
- up to `10` epochs
- early stopping on validation loss (`patience=3`)

Artifacts produced:
- `spam_classifier_tensorflow.keras`
- `tokenizer_tensorflow.pickle`

## Usage

### 1) CLI prediction

```bash
python predict.py "хочешь взять долги и кредиты без процентов"
```

Example output:

```text
--- Результат анализа ---
Сообщение:      'хочешь взять долги и кредиты без процентов'
Оценка риска:   0.5812 (58.12%)
Уровень угрозы: ⚠️ Сомнительно (требуется проверка)
-------------------------
```

### 2) Retrain model

```bash
python train.py
```

### 3) Run API server

```bash
python server.py
```

Server address:
- host: `0.0.0.0`
- port: `7503`

## API

### Health check

```bash
curl http://127.0.0.1:7503/health
```

Response:

```json
{"status":"ok"}
```

### Analyze endpoint

```bash
curl -X POST http://127.0.0.1:7503/analyze \
  -H "Content-Type: application/json" \
  -d '{"message":"хочешь взять долги и кредиты без процентов"}'
```

Response:

```json
{"risk_score":0.5811898708343506}
```

Validation behavior:
- non-JSON body → `400`
- missing `message` field → `400`
- non-string `message` → `400`
- internal prediction error → `500`

## Limitations

- Optimized for **Russian-language** spam patterns; quality may degrade on other domains/languages.
- Output is a model score, not a guaranteed fraud verdict.
- Thresholds in `predict.py` are heuristic and may need recalibration for your use case.
