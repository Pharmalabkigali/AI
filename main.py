from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class AskRequest(BaseModel):
    instrument: str
    question: str

class LoginRequest(BaseModel):
    role: str
    code: str

# LOGIN ENDPOINT
@app.post("/login")
async def login_user(data: LoginRequest):
    if data.role.lower() == "pharmalab" and data.code == "PL2025":
        return {"role": "pharmalab"}
    elif data.role.lower() == "client" and data.code == "CL2025":
        return {"role": "client"}
    else:
        return {"error": "Invalid role or access code."}


# AI ASSISTANT ENDPOINT
@app.post("/ask")
async def ask_ai(data: AskRequest):
    instrument = os.path.basename(data.instrument.lower().strip())
    question = data.question.strip()
    manual_path = f"manuals/{instrument}.txt"

    if not os.path.exists(manual_path):
        return {"answer": f"❌ Manual for '{instrument}' not found."}

    with open(manual_path, "r", encoding="utf-8") as f:
        manual_text = f.read()

    prompt = f"""You are a helpful biomedical technician assistant. Use the following service manual to answer the user's question.

Manual:
{manual_text[:8000]}

Question:
{question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600
        )
        return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"answer": f"❌ Error using Groq/Mistral: {str(e)}"}
