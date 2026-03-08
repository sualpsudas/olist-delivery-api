from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.schemas import DeliveryInput, PredictionResponse, ExplainResponse, HealthResponse
from app import model as ml


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: modeli yükle
    ml.load_model()
    yield
    # Shutdown: temizlik (gerekirse)


app = FastAPI(
    title="Olist Delivery Prediction API",
    description="""
Predicts whether an Olist order will be delivered **late or on time**.

## Endpoints

- **POST /predict** — Delivery delay prediction (On Time / Late)
- **POST /explain** — Prediction + SHAP explanation (which features matter most)
- **GET  /health**  — Model health check
- **GET  /features** — List of expected input features
""",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """API ve model durumunu kontrol et."""
    return {
        "status": "ok",
        "model": "XGBoost (olist-ml-science)",
        "features_count": len(ml.get_features()),
    }


@app.get("/features", tags=["System"])
def features():
    """Modelin beklediği feature listesini döner."""
    return {"features": ml.get_features()}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: DeliveryInput):
    """
    Sipariş özelliklerine göre teslimat gecikmesi tahmini yapar.

    - **prediction**: 0 = On Time, 1 = Late
    - **label**: "On Time" veya "Late"
    - **probability_late**: Geç kalma olasılığı (0–1)
    """
    try:
        result = ml.predict(data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse, tags=["Prediction"])
def explain(data: DeliveryInput):
    """
    Tahmin + SHAP açıklaması döner.

    - **shap_values**: Her feature'ın tahmine katkısı (pozitif = geç kalma yönünde)
    - **top_factor**: En etkili feature
    """
    try:
        pred_result = ml.predict(data.model_dump())
        expl_result = ml.explain(data.model_dump())
        return {
            **pred_result,
            **expl_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
