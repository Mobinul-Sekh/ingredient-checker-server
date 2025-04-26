from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import easyocr
import cv2
from PIL import Image
import base64
import io
import os

app = FastAPI()
load_dotenv()
reader = easyocr.Reader(['en'])
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
token = os.getenv("GITHUB_TOKEN")

client = ChatCompletionsClient(
  endpoint=endpoint,
  credential=AzureKeyCredential(token),
)

class ImageData(BaseModel):
  image: str

@app.get("/")
def greet():
  return {"message": "Hello World, server is up!"}

@app.post("/analyze-ingredients-affects")
async def ocr_image(data: ImageData):
  print("=========================analyze api hit=========================")
  if not data.image:
    raise HTTPException(status_code=400, detail="Image is required")

  try:
    image_data = base64.b64decode(data.image)
    image = Image.open(io.BytesIO(image_data))

    extractedTextData = reader.readtext(image)

    text = ""
    for data in extractedTextData:
      text += data[1] + " "

    response = client.complete(
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
            {{ingredient: data, description: data, effects: `{positive: data, negative: data}`, nutrition: data, statistics: data, regulations: data, considerations: data}}

            Here are the ingredients: {text}"""
          )
      ],
      temperature=1,
      top_p=1,
      model=model
    )

    return {
      "success": True,
      "message": "analyzed successfully",
      "data": response.choices[0].message.content,
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
