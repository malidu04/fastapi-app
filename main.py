from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os

# Initialize FastAPI app
app = FastAPI()

# Define file paths (Ensure models are inside the 'app' directory)
MODEL_PATH = os.path.join(os.getcwd(), "personality_prediction_ann_lstm.h5")
SCALER_PATH = os.path.join(os.getcwd(), "scaler.pkl")
ENCODER_PATH = os.path.join(os.getcwd(), "label_encoder.pkl")

# Load the model and utilities
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Define the input structure
class QuizInput(BaseModel):
    quiz_responses: list

@app.post("/predict_personality/")
async def predict_personality(data: QuizInput):
    try:
        # Preprocess input
        new_quiz_responses = np.array(data.quiz_responses).reshape(1, -1)
        new_quiz_responses_scaled = scaler.transform(new_quiz_responses)

        # Reshape for LSTM
        new_quiz_responses_scaled = new_quiz_responses_scaled.reshape(1, 1, new_quiz_responses_scaled.shape[1])

        # Predict
        prediction = model.predict(new_quiz_responses_scaled)
        predicted_label = np.argmax(prediction, axis=1)
        predicted_personality = label_encoder.inverse_transform(predicted_label)

        return {"predicted_personality": predicted_personality[0]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")
