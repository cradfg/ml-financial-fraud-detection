from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import TransactionInput, PredictionOutput
from app.model import predict

app = FastAPI(
    title="ML Financial Fraud Detection API",
    description="LightGBM-based fraud detection trained on IEEE-CIS dataset",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",         # live server local dev
        "http://127.0.0.1:5500",
        "https://github.com/cradfg", # My github page
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# Routes 

@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud Detection API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict_fraud(transaction: TransactionInput):
    try:
        result = predict(transaction.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#Entry point 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)