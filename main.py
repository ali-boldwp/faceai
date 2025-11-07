from fastapi import FastAPI
from app.routers import face_routes

app = FastAPI(title="Face Morphology API")

app.include_router(face_routes.router)

@app.get("/")
async def root():
    return {"message": "API is running 🚀"}
