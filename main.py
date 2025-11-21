from fastapi import FastAPI
from app.routers import face_routes
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Face Morphology API")

origins = [
    "http://localhost:3000",
    "https://web--main--faceaimern--ali.code.devregion.com",
    "https://faceai.pwtech.ro",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,      # only if you actually use cookies/auth in frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(face_routes.router)

@app.get("/")
async def root():
    return {"message": "API is running 🚀"}
