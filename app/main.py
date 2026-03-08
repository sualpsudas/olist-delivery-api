import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.schemas import DeliveryInput, PredictionResponse, ExplainResponse, HealthResponse
from app.auth import verify_api_key
from app import model as ml

load_dotenv()

# Rate limiter — istek başına IP adresine göre sınırlar
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml.load_model()
    yield


app = FastAPI(
    title="Olist Delivery Prediction API",
    description="""
Predicts whether an Olist order will be delivered **late or on time**.

## Authentication
All prediction endpoints require an `X-API-Key` header:
```
X-API-Key: your-api-key
```

## Rate Limits
- `/predict` → 10 requests/minute
- `/explain` → 5 requests/minute

## Endpoints
- **POST /predict** — Delivery delay prediction
- **POST /explain** — Prediction + SHAP explanation
- **GET  /health**  — Health check (no auth required)
- **GET  /features** — Feature list (no auth required)
""",
    version="2.0.0",
    lifespan=lifespan,
)

# Rate limit hata handler'ı kaydet
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Public endpoints (auth yok) ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """API ve model durumunu kontrol et. Auth gerektirmez."""
    return {
        "status": "ok",
        "model": "XGBoost (olist-ml-science)",
        "features_count": len(ml.get_features()),
    }


@app.get("/features", tags=["System"])
def features():
    """Modelin beklediği feature listesi. Auth gerektirmez."""
    return {"features": ml.get_features()}


# ── Protected endpoints (API key + rate limit) ────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(os.getenv("RATE_LIMIT_PREDICT", "10/minute"))
def predict(request: Request, data: DeliveryInput):
    """
    Sipariş özelliklerine göre teslimat gecikmesi tahmini.

    **Auth:** X-API-Key header gerekli
    **Rate limit:** 10 istek/dakika
    """
    try:
        return ml.predict(data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["Prediction"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(os.getenv("RATE_LIMIT_EXPLAIN", "5/minute"))
def explain(request: Request, data: DeliveryInput):
    """
    Tahmin + SHAP açıklaması.

    **Auth:** X-API-Key header gerekli
    **Rate limit:** 5 istek/dakika
    """
    try:
        pred = ml.predict(data.model_dump())
        expl = ml.explain(data.model_dump())
        return {**pred, **expl}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
