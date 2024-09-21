from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the data for safe area prediction
data = pd.read_csv("2018 Victims of Rape.csv")

# Prepare the data for the model
X = data[['Cases Reported', 'Child Victims of Rape (Below 18 Yrs) - Total Girl/Child Victims', 
          'Women Victims of Rape (Above 18 Yrs) - Total Women/Adult Victims']]
y = data['State/UT ']

# Train a RandomForest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to safely convert numpy types to native Python types
def convert_to_python_type(value):
    if isinstance(value, (np.generic, np.int64, np.float64)):
        return value.item()
    return value

# Chatbot setup
emotions = [
    ("I feel happy today", "joy"),
    ("I'm so excited", "joy"),
    ("I'm feeling sad", "sadness"),
    ("I'm really upset", "anger"),
    ("I'm worried about tomorrow", "anxiety"),
    ("I feel so relaxed", "calm"),
    ("Hello! How I can help you?", "Hi"),
   
]

# Prepare the data for chatbot
X_chat, y_chat = zip(*emotions)
vectorizer = CountVectorizer()
X_chat = vectorizer.fit_transform(X_chat)

# Train a simple Naive Bayes classifier for chatbot
clf = MultinomialNB()
clf.fit(X_chat, y_chat)

# Save the model and vectorizer
joblib.dump(clf, 'emotion_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

# Pydantic models for input validation
class PredictionInput(BaseModel):
    state: str

class ChatInput(BaseModel):
    message: str

# Safe area prediction endpoint
@app.post("/predict")
async def predict(input: PredictionInput):
    try:
        state_data = data[data['State/UT '] == input.state]
        
        if state_data.empty:
            raise HTTPException(status_code=404, detail="State not found")

        state_data = state_data.iloc[0]

        features = [[state_data['Cases Reported'], 
                     state_data['Child Victims of Rape (Below 18 Yrs) - Total Girl/Child Victims'],
                     state_data['Women Victims of Rape (Above 18 Yrs) - Total Women/Adult Victims']]]

        prediction = model.predict(features)

        return {"safety": prediction[0] == input.state, "danger": prediction[0] != input.state}

    except IndexError:
        raise HTTPException(status_code=404, detail="State data not found")

# Get state data endpoint
@app.get("/state_data/{state}")
async def get_state_data(state: str):
    try:
        state_data = data[data['State/UT '] == state]
        
        if state_data.empty:
            raise HTTPException(status_code=404, detail="State not found")

        state_data = state_data.iloc[0]

        return {
            "Cases Reported": convert_to_python_type(state_data['Cases Reported']),
            "Child Victims": convert_to_python_type(state_data['Child Victims of Rape (Below 18 Yrs) - Total Girl/Child Victims']),
            "Adult Victims": convert_to_python_type(state_data['Women Victims of Rape (Above 18 Yrs) - Total Women/Adult Victims'])
        }
    
    except IndexError:
        raise HTTPException(status_code=404, detail="State data not found")

@app.post("/chat")
async def chat(input: ChatInput):
    print(f"Received message: {input.message}")  # Log received message
    user_message = input.message
    
    try:
        # Preprocess the input
        user_vector = vectorizer.transform([user_message])
        
        # Make a prediction
        emotion = clf.predict(user_vector)[0]
        print(f"Detected emotion: {emotion}")  # Log detected emotion
        
        # Generate a response based on the predicted emotion
        responses = {
            "joy": "I'm glad you're feeling happy! What's making you feel this way?",
            "sadness": "I'm sorry to hear you're feeling sad. Would you like to talk about it?",
            "anger": "I can sense that you're angry. Take a deep breath. Want to discuss what's bothering you?",
            "anxiety": "It sounds like you're feeling anxious. Remember, it's okay to feel this way. Can I help you work through your concerns?",
            "calm": "It's great that you're feeling calm. What's contributing to your sense of peace?"
        }
        
        response = responses.get(emotion, "I see. Can you tell me more about how you're feeling?")
        print(f"Sending response: {response}")  # Log response
        
        return {"response": response, "detected_emotion": emotion}
    except Exception as e:
        print(f"Error processing message: {str(e)}")  # Log any errors
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)