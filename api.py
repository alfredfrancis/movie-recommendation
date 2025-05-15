import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import uvicorn
from typing import List, Dict, Any
import time
import logging
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

model = None
tokenizer = None

device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"

logger.info(f"Using device: {device_name}")

device = torch.device(device_name)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    logger.info("Loading BERT model and tokenizer")
    start_time = time.time()
    
    global model, tokenizer
    try:
        model = BertForSequenceClassification.from_pretrained("./bert-imdb")
        tokenizer = BertTokenizer.from_pretrained("./bert-imdb")
        model.eval() 
        model.to(device)
            
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    yield
    
    logger.info("Shutting down and releasing resources")
    model = None
    tokenizer = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

app = FastAPI(
    title="Sentiment Analysis API",
    description="High-throughput API for BERT-based sentiment analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class SentimentRequest(BaseModel):
    text: str

class BatchSentimentRequest(BaseModel):
    texts: List[str]

# Response models
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

class BatchSentimentResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time: float

def predict_sentiment_with_score(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    
    return {
        "sentiment": "positive" if prediction == 1 else "negative",
        "confidence": confidence
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request to {request.url.path} completed in {process_time:.4f}s")
    return response

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    try:
        result = predict_sentiment_with_score(request.text)
        return {
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch(request: BatchSentimentRequest):
    start_time = time.time()
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="Empty batch provided")
    
    if len(request.texts) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 100")
    
    try:
        results = []
        for text in request.texts:
            if text.strip():
                result = predict_sentiment_with_score(text)
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": result["sentiment"],
                    "confidence": result["confidence"]
                })
            else:
                results.append({
                    "text": "",
                    "sentiment": "unknown",
                    "confidence": 0.0,
                    "error": "Empty text provided"
                })
        
        process_time = time.time() - start_time
        
        return {
            "results": results,
            "processing_time": process_time
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9005,
        # workers=min(os.cpu_count() or 1, 4),  # Adjust workers based on available CPUs
        log_level="info"
    )