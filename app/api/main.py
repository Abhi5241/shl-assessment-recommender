from fastapi import FastAPI
from pydantic import BaseModel

from app.services.recommendation_service import RecommendationService
from app.core.logging import get_logger

logger = get_logger()

app = FastAPI(
    title="SHL Assessment Recommendation API",
    version="1.0"
)

service = RecommendationService()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def health():
    return {"status": "API running"}


@app.post("/recommend")
def recommend(request: QueryRequest):
    logger.info(f"Received query: {request.query}")
    result = service.recommend(request.query)
    return result