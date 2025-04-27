from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import easyocr
from PIL import Image
import base64
import io
import os
from db import engine, Base, get_db, SessionLocal
from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import Session


# task model interface
class Task(Base):
  __tablename__ = "task"

  id = Column(Integer, primary_key=True, index=True)
  status = Column(String, default="processing")
  result = Column(String, nullable=True)

class ImageData(BaseModel):
  image: str


# create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
load_dotenv()
reader = easyocr.Reader(['en'], model_storage_directory='/tmp/easyocr')

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
token = os.getenv("GITHUB_TOKEN")
PORT = os.getenv("PORT")

client = ChatCompletionsClient(
  endpoint=endpoint,
  credential=AzureKeyCredential(token),
)


@app.get("/")
def greet():
  return {"message": "Hello World, server is up!"}


@app.post("/analyze-ingredients-affects")
async def ocr_image(
  data: ImageData,
  background_task: BackgroundTasks,
  db: Session = Depends(get_db)
):
  if not data.image:
    raise HTTPException(status_code=400, detail="Image is required")

  new_task = Task()
  db.add(new_task)
  db.commit()
  db.refresh(new_task)

  background_task.add_task(process_ocr_image, data.image, new_task.id)

  return {"success": True, "message": "Processing started...", "task_id": str(new_task.id)}
  

@app.get("/get-analysis-result/{task_id}")
def get_task(task_id: int, db: Session = Depends(get_db)):
  task = db.query(Task).filter(Task.id == task_id).first()
  if not task:
      raise HTTPException(status_code=404, detail="Task not found")
  return {
    "id": task.id,
    "status": task.status,
    "result": task.result
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

def process_ocr_image(
  image_b64: str,
  task_id: int
):
  db = SessionLocal()

  try:
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    image = compress_image(image)
    
    extractedTextData = reader.readtext(image)

    text = ""
    for data in extractedTextData:
      text += data[1] + " "

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

              Strictly return the output in this array of objects format (no extra text):
              {{ingredient: data, description: data, effects: `object with positive: data, negative: data`, nutrition: data, statistics: data, regulations: data, considerations: data}}

              Here are the ingredients: {text}"""
            )
        ],
        temperature=1,
        top_p=1,
        model=model
      )

    response = retry_call(call_openai)

    task = db.query(Task).filter(Task.id == task_id).first()
    task.status = "completed"
    task.result = response.choices[0].message.content
    db.commit()
    
    print("response: ", response.choices[0].message.content)

  except Exception as e:
    task = db.query(Task).filter(Task.id == task_id).first()
    task.status = "failed"
    task.result = str(e)
    db.commit()
  
  finally:
    db.close()
