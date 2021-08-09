from fastapi import FastAPI
from pydantic import BaseModel
from twitter import get_tweets, create_api
from utils import create_data
from nn import MLPerceptron
from dotenv import load_dotenv
from transformers import transform_data, transform_new
import numpy as np


load_dotenv()

class Body(BaseModel):
  name_1: str
  name_2: str
  text: str
    
class Payload(BaseModel):
  code: str = "None"
  message: str = "None"

app = FastAPI()

@app.get("/")
async def root():
  return {"message": "Nothing to see here."}

@app.post("/predict")
async def predict(body: Body):
  name_1 = body.name_1
  name_2 = body.name_2
  text = np.array([body.text])

  payload = Payload()
  # Gather tweets from given users
  try:
    api = create_api()
    X, y = get_tweets(name_1, name_2, api)
    x_load = True
  except Exception as e:
    payload.code = "error"
    payload.message = e
    

  if payload.code != "error":
    X, y = create_data(X, y)
    # Transform X to LSA vectors
    vect, svd, X = transform_data(X)
    model = MLPerceptron(X.shape[1], hidden_dim=20, lr=0.01)
    model.train(X, y, 500)
    # Predict with trained model

    pred_text = transform_new(vect, svd, text)
    prediction = model.predict(pred_text)
    payload.code = "success"
    payload.message = str(prediction[0][0])

  return payload

@app.post("/pred_test")
async def test(body: Body):
  return body
