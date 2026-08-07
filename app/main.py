from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core import logger
from app.routes import router as predict_route
from app.routes import router as predict_overall_route
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_route, prefix="/route")
app.include_router(predict_overall_route, prefix="/route")




@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Hello, World!"}