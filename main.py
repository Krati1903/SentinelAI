"""
Entrypoint for the AI Interview Proctoring System.
Run with: python main.py
"""

import uvicorn
from fastapi import FastAPI

from app.api.routes import router
from app.logger import logger

app = FastAPI(
    title="AI Interview Proctoring System",
    description=(
        "Real-time interview proctoring API. Detects face visibility, "
        "multiple persons, head pose, gaze direction, and prohibited "
        "objects (phones, books, tablets, laptops) using MediaPipe and YOLOv8."
    ),
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
def on_startup():
    logger.info("AI Interview Proctoring System starting up...")


@app.on_event("shutdown")
def on_shutdown():
    logger.info("AI Interview Proctoring System shutting down...")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
