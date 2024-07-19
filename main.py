from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from src.recommender_service import RecommenderService
from src.explanation_service import ExplanationService
from typing import List

class Recommendation(BaseModel):
    recipe_id: str
    probability: float

app = FastAPI(
    title="Recommendation and explanation model",
    version=1.0,
    description="API for recommendation and explanation model"
)


@app.get("/")
def root():
    return {"message": "Welcome to the recommendation and explanation model API"}

@app.get("/healthcheck/")
def healthcheck():
    return {"status": "OK"}

@app.post("/recommend/", response_model=List[Recommendation])
async def predict_items(data: Request, num_recommendations: int):
    try:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while predicting items: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", 
                host="localhost", 
                port=8500,
                reload=True)