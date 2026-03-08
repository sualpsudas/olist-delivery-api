from pydantic import BaseModel, Field
from typing import Optional


class DeliveryInput(BaseModel):
    estimated_days: int = Field(..., ge=1, le=90, description="Estimated delivery days by seller")
    item_count: int = Field(..., ge=1, le=20, description="Number of items in order")
    total_price: float = Field(..., gt=0, description="Total order price (BRL)")
    freight_value: float = Field(..., ge=0, description="Freight cost (BRL)")
    product_weight_g: float = Field(..., gt=0, description="Product weight in grams")
    product_volume: float = Field(..., gt=0, description="Product volume (cm³)")
    payment_installments: float = Field(..., ge=1, le=24, description="Number of payment installments")
    same_state: int = Field(..., ge=0, le=1, description="1 if seller and customer in same state")
    purchase_dow: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    purchase_month: int = Field(..., ge=1, le=12, description="Month of purchase")
    purchase_hour: int = Field(..., ge=0, le=23, description="Hour of purchase")
    payment_type: int = Field(..., ge=0, le=3, description="Payment type encoded (0=boleto,1=credit,2=debit,3=voucher)")
    customer_state: int = Field(..., ge=0, le=26, description="Customer state encoded (0-26)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_late: float
    probability_on_time: float


class ExplainResponse(BaseModel):
    prediction: int
    label: str
    probability_late: float
    base_value: float
    shap_values: dict
    top_factor: str


class HealthResponse(BaseModel):
    status: str
    model: str
    features_count: int
