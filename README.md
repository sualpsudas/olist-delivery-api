# Olist Delivery Prediction API

![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi) ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange) ![SHAP](https://img.shields.io/badge/SHAP-explainability-red)

REST API that predicts whether an Olist order will be delivered **late or on time**, powered by an XGBoost model trained on 77K orders. Includes SHAP-based explanations.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Model health check |
| `GET`  | `/features` | List of expected input features |
| `POST` | `/predict` | Delivery delay prediction |
| `POST` | `/explain` | Prediction + SHAP explanation |
| `GET`  | `/docs` | Interactive Swagger UI |

---

## Quick Start

### Local

```bash
# 1. Clone
git clone https://github.com/sualpsudas/olist-delivery-api.git
cd olist-delivery-api

# 2. Install
conda activate ai
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env: set your API_KEYS

# 4. Run
uvicorn app.main:app --reload --port 8000
```

API → http://localhost:8000
Swagger → http://localhost:8000/docs

### Docker

```bash
# Build & run
docker compose up --build

# Or with custom keys
docker run -e API_KEYS=mykey -p 8000:8000 olist-delivery-api
```

---

## Authentication

All prediction endpoints require an `X-API-Key` header:

```bash
# Set in .env
API_KEYS=secret-key-123,dev-key-456

# Use in requests
curl -H "X-API-Key: secret-key-123" ...
```

| Status | Meaning |
|--------|---------|
| `401` | Header missing |
| `403` | Invalid key |

## Rate Limiting

| Endpoint | Limit |
|----------|-------|
| `/predict` | 10 req/min |
| `/explain` | 5 req/min |

Exceeding the limit returns `429 Too Many Requests`.

---

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: secret-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "estimated_days": 10,
    "item_count": 1,
    "total_price": 150.0,
    "freight_value": 18.5,
    "product_weight_g": 500.0,
    "product_volume": 3000.0,
    "payment_installments": 1.0,
    "same_state": 0,
    "purchase_dow": 2,
    "purchase_month": 6,
    "purchase_hour": 14,
    "payment_type": 1,
    "customer_state": 9
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Late",
  "probability_late": 0.7315,
  "probability_on_time": 0.2685
}
```

---

## SHAP Explanation

```bash
curl -X POST http://localhost:8000/explain ...
```

```json
{
  "prediction": 1,
  "label": "Late",
  "probability_late": 0.7315,
  "base_value": -0.001,
  "shap_values": {
    "customer_state": 1.1065,
    "estimated_days": 0.8127,
    "purchase_month": -1.0619,
    ...
  },
  "top_factor": "customer_state"
}
```

Positive SHAP → pushes toward **Late**. Negative → toward **On Time**.

---

## Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `estimated_days` | int | Seller's estimated delivery days |
| `item_count` | int | Items in order |
| `total_price` | float | Order total (BRL) |
| `freight_value` | float | Shipping cost (BRL) |
| `product_weight_g` | float | Weight in grams |
| `product_volume` | float | Volume in cm³ |
| `payment_installments` | float | Number of installments |
| `same_state` | int | 1 if seller & customer in same state |
| `purchase_dow` | int | Day of week (0=Mon) |
| `purchase_month` | int | Month (1–12) |
| `purchase_hour` | int | Hour of day (0–23) |
| `payment_type` | int | 0=boleto, 1=credit, 2=debit, 3=voucher |
| `customer_state` | int | State encoded (0–26) |

---

## Model

- **Algorithm:** XGBoost Classifier
- **Training data:** 77,176 Olist orders
- **Target:** `delayed` (0 = On Time, 1 = Late)
- **Weighted F1:** 0.81
- **Source:** [olist-ml-science](https://github.com/sualpsudas/olist-ml-science)

---

## Project Structure

```
olist-delivery-api/
├── app/
│   ├── main.py       # FastAPI app, routes, rate limiting
│   ├── auth.py       # API key authentication
│   ├── model.py      # Model loading, predict, explain
│   └── schemas.py    # Pydantic input/output models
├── models/
│   └── xgb_model.pkl # Trained XGBoost model
├── tests/
│   └── test_api.py   # 7 pytest tests (auth + prediction)
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Tests

```bash
pytest tests/ -v
# 7 passed
```

Covers: health, features, auth (no key, wrong key), predict, validation error, explain.
