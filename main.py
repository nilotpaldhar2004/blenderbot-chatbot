# main.py — BlenderBot Conversational AI — FastAPI Server
#
# Install:  pip install -r requirements.txt
# Run:      uvicorn main:app --host 0.0.0.0 --port 8000
# Docs:     http://localhost:8000/docs   (Swagger UI, auto-generated)
# Test:     curl -X POST http://localhost:8000/chat \
#             -H "Content-Type: application/json" \
#             -d '{"message": "Hello!"}'

import logging
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BlenderBot Conversational AI",
    description="Fine-tuned facebook/blenderbot-400M-distill on blended_skill_talk dataset",
    version="1.0.0",
)

# Allow all origins so the HTML frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend HTML file at /
app.mount("/static", StaticFiles(directory="."), name="static")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR      = "nilotpaldhar2004/blenderbot-chatbot"   
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEP            = "  "                        # BlenderBot's conversation separator (double space)
MAX_CTX_LEN    = 128
MAX_NEW_TOKENS = 60

# ── Load model once at startup ────────────────────────────────────────────────
logger.info(f"Loading model from '{MODEL_DIR}' on {DEVICE} ...")
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_DIR)
model     = BlenderbotForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
logger.info("Model loaded and ready.")


# ── Request / Response schemas ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: List[str] = []   # alternating [user_msg, bot_msg, user_msg, ...]

class ChatResponse(BaseModel):
    response: str

class HealthResponse(BaseModel):
    status: str
    device: str
    model: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
def serve_frontend():
    """Serve the HTML chat UI."""
    return FileResponse("index.html")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Quick health check — confirms the model is loaded."""
    return HealthResponse(status="ok", device=str(DEVICE), model=MODEL_DIR)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Generate a conversational reply.

    Send history=[] for a new conversation.
    Append previous [user_msg, bot_msg] pairs to history for multi-turn context.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message field cannot be empty")

    # BlenderBot expects turns joined with double-space separator
    context = SEP.join(req.history + [req.message.strip()])

    inputs = tokenizer(
        context,
        return_tensors="pt",
        max_length=MAX_CTX_LEN,
        truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            min_length=10,
            num_beams=3,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if not response:
        response = "I am not sure how to respond to that."

    logger.info(f"User: {req.message[:60]!r}  |  Bot: {response[:60]!r}")
    return ChatResponse(response=response)


@app.delete("/history")
def clear_history():
    """
    The server is stateless — history lives on the client side.
    Call /chat with history=[] to start a fresh conversation.
    """
    return {"detail": "Pass history=[] in your next /chat request to reset."}
