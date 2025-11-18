import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai


API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
model = genai.GenerativeModel(MODEL_NAME)

app = FastAPI(title="Gemini AI Agent")

class UserRequest(BaseModel):
    message: str

@app.post("/agent")
def agent_endpoint(req: UserRequest):
    try:
        response = model.generate_content(req.message)
        reply = getattr(response, "text", None) or str(response)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    