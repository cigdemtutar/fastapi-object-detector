from fastapi import FastAPI, File, HTTPException, UploadFile

from app.model import Detector

app = FastAPI(title="Object Detector API")
detector = Detector("models/detector_weights.pth", "models/classes.json")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return detector.predict(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
