from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import random

router = APIRouter()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@router.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr   = np.frombuffer(contents, np.uint8)
        img      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces  = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return JSONResponse(status_code=200, content={
                "success":       False,
                "face_detected": False,
                "message":       "No face detected. Please look directly at the camera.",
            })

        # Basic brightness/contrast analysis as proxy for engagement
        x, y, w, h = faces[0]
        face_roi   = gray[y:y+h, x:x+w]
        brightness = float(np.mean(face_roi))
        contrast   = float(np.std(face_roi))

        # Normalize scores
        engagement_score = round(min(brightness / 25.5, 10), 1)
        concern_score    = round(max(0, 10 - contrast / 12.75), 1)

        return {
            "success":          True,
            "face_detected":    True,
            "dominant_emotion": "neutral",
            "emotions": {
                "happy":   round(engagement_score * 8, 1),
                "neutral": round(100 - engagement_score * 8, 1),
                "sad":     round(concern_score * 5, 1),
                "fear":    0.0,
                "angry":   0.0,
                "surprise":0.0,
            },
            "engagement_score": engagement_score,
            "concern_score":    concern_score,
            "face_count":       len(faces),
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success":       False,
            "face_detected": False,
            "error":         str(e),
        })