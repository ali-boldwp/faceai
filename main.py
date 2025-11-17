from fastapi import FastAPI
from app.routers import face_routes
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Face Morphology API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://faceai.pwtech.ro",
        "https://preview--main--face-ai--zaman.code.devregion.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(face_routes.router)

@app.get("/")
async def root():
    return {"message": "API is running 🚀"}
