from pydantic import BaseModel, Field
from typing import Optional


class TransactionInput(BaseModel):
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD: Optional[str] = Field(None, description="Product code (W, H, C, S, R)")
    card4: Optional[str] = Field(None, description="Card network (visa, mastercard, etc)")
    card6: Optional[str] = Field(None, description="Card type (debit, credit)")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    DeviceType: Optional[str] = Field(None, description="Device type (desktop, mobile)")
    DeviceInfo: Optional[str] = Field(None, description="Device info string")
    addr1: Optional[float] = Field(None, description="Billing address zip code")
    addr2: Optional[float] = Field(None, description="Billing address country code")
    dist1: Optional[float] = Field(None, description="Distance between addresses")
    C1:  Optional[float] = Field(None, description="Count feature C1")
    C2:  Optional[float] = Field(None, description="Count feature C2")
    C6:  Optional[float] = Field(None, description="Count feature C6")
    C11: Optional[float] = Field(None, description="Count feature C11")
    C13: Optional[float] = Field(None, description="Count feature C13")
    C14: Optional[float] = Field(None, description="Count feature C14")
    D1:  Optional[float] = Field(None, description="Timedelta feature D1")
    D4:  Optional[float] = Field(None, description="Timedelta feature D4")
    D10: Optional[float] = Field(None, description="Timedelta feature D10")
    D15: Optional[float] = Field(None, description="Timedelta feature D15")
    M4:  Optional[str]   = Field(None, description="Match feature M4")
    M6:  Optional[str]   = Field(None, description="Match feature M6")

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 150.0,
                "ProductCD":      "W",
                "card4":          "visa",
                "card6":          "debit",
                "P_emaildomain":  "gmail.com",
                "R_emaildomain":  "gmail.com",
                "DeviceType":     "desktop",
                "DeviceInfo":     "Windows",
                "addr1":          315.0,
                "addr2":          87.0,
                "dist1":          19.0,
                "C1":  1.0,
                "C2":  1.0,
                "C6":  1.0,
                "C11": 1.0,
                "C13": 1.0,
                "C14": 1.0,
                "D1":  14.0,
                "D4":  0.0,
                "D10": 0.0,
                "D15": 0.0,
                "M4":  "M2",
                "M6":  "T",
            }
        }


class PredictionOutput(BaseModel):
    fraud_probability: float = Field(..., description="Model confidence score (0–1)")
    is_fraud:          bool  = Field(..., description="Verdict at tuned threshold")
    risk_level:        str   = Field(..., description="LOW / MEDIUM / HIGH")
    threshold_used:    float = Field(..., description="Decision threshold from training")