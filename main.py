from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.predict import router as predict_router
from routes.analytics import router as analytics_router
from routes.face import router as face_router
from routes.image_predict import router as image_predict_router

app = FastAPI(title="ASD Screening API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://e745-49-205-37-117.ngrok-free.app",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(face_router, prefix="/api")
app.include_router(image_predict_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "ASD Screening API is running ✅"}