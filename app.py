from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("emotion_model.joblib")

label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

class TextInput(BaseModel):
    text: str

# FastAPI app
app = FastAPI()

@app.post("/predict")
def predict_emotion(input: TextInput):
    prediction = model.predict([input.text])[0]
    emotion = label_map[int(prediction)]
    return {"text": input.text, "predicted_label": int(prediction), "emotion": emotion}
@app.get("/")
def read_root():
    return {"message": "Welcome to the Emotion Prediction API. Use the /predict endpoint to get predictions."}
