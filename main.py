from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend domain if needed
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

    manual_path = f"manuals{instrument}.txt"

    if not os.path.exists(manual_path):
        return {"answer": f"Manual for '{instrument}' not found."}

    with open(manual_path, "r", encoding="utf-8") as f:
        manual_text = f.read()

    prompt = f"""
You are a helpful biomedical engineering assistant. Use the following service manual to answer the user's question accurately.

Service Manual Content:
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
        client = openai.OpenAI()  # initialize the client

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=300,
    temperature=0.4,
)

return {"answer": response.choices[0].message.content.strip()}

    except Exception as e:
        return {"answer": f"❌ Error calling OpenAI: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <h2>✅ PharmaLab AI Assistant Backend is Running</h2>
    <p>You can send a POST request to <code>/ask</code> with instrument and question.</p>
    <p>Example JSON body:</p>
    <pre>{
  "instrument": "humacount5",
  "question": "How to reset the device?"
}</pre>
    """
