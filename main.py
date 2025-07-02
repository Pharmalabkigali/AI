from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai

# Load environment variables
load_dotenv()

# Initialize OpenAI client with your API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI app
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use your domain here for security in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class AskRequest(BaseModel):
    instrument: str
    question: str

# Define route
@app.post("/ask")
async def ask_ai(data: AskRequest):
    instrument = data.instrument.lower().strip()
    question = data.question.strip()
    manual_path = f"manuals/{instrument}.txt"

    if not os.path.exists(manual_path):
        return {"answer": f"❌ Manual for '{instrument}' not found."}

    with open(manual_path, "r", encoding="utf-8") as f:
        manual_text = f.read()

    # Prompt for AI
    prompt = f"""
You are a helpful biomedical engineering assistant. Use the following service manual to answer the user's question.

Service Manual:
\"\"\"
{manual_text[:3000]}
\"\"\"

User Question:
\"\"\"
{question}
\"\"\"

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.4
        )
        return {"answer": response.choices[0].message.content.strip()}

    except Exception as e:
        return {"answer": f"❌ Error calling OpenAI: {str(e)}"}
