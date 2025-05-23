from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from PIL import Image
import base64
import io
import os
from db import engine, Base, get_db, SessionLocal
from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import Session
import requests
import pytesseract
import time

# task model interface
class Task(Base):
  __tablename__ = "task"

  id = Column(Integer, primary_key=True, index=True)
  status = Column(String, default="processing")
  result = Column(String, nullable=True)
  created_at = Column(String, nullable=True)
  updated_at = Column(String, nullable=True)

class ImageData(BaseModel):
  image: str


# create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

load_dotenv()

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

token = os.getenv("GITHUB_TOKEN")
PORT = os.getenv("PORT")
imgur_client_id = os.getenv("IMGUR_CLIENT_ID")

client = ChatCompletionsClient(
  endpoint=endpoint,
  credential=AzureKeyCredential(token),
)


@app.get("/")
def greet():
  print("===========================home route hit===========================")
  return {"message": "Hello World, server is up!"}


@app.post("/analyze-ingredients-affects")
async def ocr_image(
  data: ImageData,
  background_task: BackgroundTasks,
  db: Session = Depends(get_db)
):
  print("===========================analyze-ingredients-affects hit===========================")
  if not data.image:
    raise HTTPException(status_code=400, detail="Image is required")

  new_task = Task()
  new_task.created_at = str(int(round(time.time() * 1000)))
  new_task.updated_at = new_task.created_at
  db.add(new_task)
  db.commit()
  db.refresh(new_task)

  background_task.add_task(process_ocr_image, data.image, new_task.id)

  return {"success": True, "message": "Processing started...", "task_id": new_task.id}
  

@app.get("/get-analysis-result/{task_id}")
def get_task(task_id: int, db: Session = Depends(get_db)):
  print("===========================get-analysis-result hit===========================")
  task = db.query(Task).filter(Task.id == task_id).first()
  if not task:
    raise HTTPException(status_code=404, detail="Task not found")
  return {
    "id": task.id,
    "status": task.status,
    "result": str(task.result)
  }

@app.get("/get-all-analysis-results")
def get_all_results(db: Session = Depends(get_db)):
  print("===========================get-all-results hit===========================")
  tasks = db.query(Task).filter(Task.status == "completed").all()
  if not tasks:
    raise HTTPException(status_code=404, detail="Results not found")
  return {
    "message": "Results fetched successfully",
    "success": True,
    "data": tasks
  }

def compress_image(image: Image.Image, max_size=(1024, 1024)) -> Image.Image:
  """Resize while maintaining aspect ratio."""
  image.thumbnail(max_size, Image.Resampling.LANCZOS)
  return image

def retry_call(func, retries=3):
  """Simple retry wrapper"""
  for attempt in range(retries):
    try:
      return func()
    except Exception as e:
      if attempt == retries - 1:
        raise e

async def upload_image_to_imgur(image_base64: str) -> str:
  """Upload base64 image to imgur anonymously."""
  url = "https://api.imgur.com/3/image"
  headers = {
    "Authorization": f"Client-ID {imgur_client_id}"
  }
  payload = {
    "image": image_base64,
    "type": "base64"
  }
  response = requests.post(url, headers=headers, data=payload)
  if response.status_code == 200:
    return response.json()["data"]["link"]
  else:
    raise Exception(f"Imgur upload failed: {response.text}")

async def process_ocr_image(
  image_b64: str,
  task_id: int
):
  db = SessionLocal()

  try:
    # uploaded_url = await upload_image_to_imgur(image_b64)
    # print("uplaoded_url ->", uploaded_url)
    # image = compress_image(image)

    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    text = pytesseract.image_to_string(image)

    print("text ->", text)

    def call_openai():
      return client.complete(
        messages=[
          SystemMessage(""),
          UserMessage(
            f"""Analyze the following list of ingredients and return structured, evidence-based health insights. For each ingredient, include:
              - A short scientific or common description
              - Health effects (positive and negative, including known side effects or toxicity)
              - Nutritional impact (e.g., caloric value, sugar/sodium/fat content, glycemic index)
              - Known population-level statistics (e.g., % sensitive/allergic, studies on long-term health impact)
              - Regulatory notes (e.g., FDA, WHO, EU classifications or warnings)
              - Any relevant dietary/medical considerations (e.g., allergens, carcinogens, preservatives, banned substances)
              - The ingredient field must content only the ingredient name

              Strictly return an array of objects:
              {{ingredient: ..., description: ..., effects: {{positive: ..., negative: ...}}, nutrition: ..., statistics: ..., regulations: ..., considerations: ...}}

              Return empty array if there is no ingredient to be found.

              Here are the ingredients: {text}"""
            )
        ],
        temperature=0.7,
        top_p=1,
        model=model
      )

    response = retry_call(call_openai)
    print("response: ", response.choices[0].message.content)

    task = db.query(Task).filter(Task.id == task_id).first()
    task.status = "completed"
    task.updated_at = str(int(round(time.time() * 1000)))
    task.result = response.choices[0].message.content
    db.commit()

  except Exception as e:
    task = db.query(Task).filter(Task.id == task_id).first()
    task.status = "failed"
    task.result = str(e)
    db.commit()
  
  finally:
    db.close()
