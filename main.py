from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from groq import Groq

# Load environment variables
load_dotenv()

# Get API Key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    instrument: str
    question: str

@app.post("/ask")
async def ask_ai(data: AskRequest):
    instrument = data.instrument.lower().strip()
    question = data.question.strip()
    manual_path = f"manuals/{instrument}.txt"

    if not os.path.exists(manual_path):
        return {"answer": f"❌ Manual for '{instrument}' not found."}

    with open(manual_path, "r", encoding="utf-8") as f:
        manual_text = f.read()

    prompt = f"""
You are a helpful biomedical technician assistant. Use the following service manual to answer the user's question.

Manual:
\"\"\"
{manual_text[:3000]}
\"\"\"

Question:
\"\"\"
{question}
\"\"\"

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # ✅ supported Groq model

            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300
        )
        return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"answer": f"❌ Error using Groq/Mistral: {str(e)}"}
