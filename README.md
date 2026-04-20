# BlenderBot Conversational AI

Fine-tuned `facebook/blenderbot-400M-distill` on the `blended_skill_talk` dataset for open-domain multi-turn conversation.

---

## Model

| Property | Value |
|---|---|
| Base model | `facebook/blenderbot-400M-distill` |
| Parameters | 364.8M |
| Dataset | `blended_skill_talk` (Facebook AI Research) |
| Architecture | Encoder-Decoder (seq2seq) |
| Training metric | Perplexity (val PPL ~14) |
| Framework | PyTorch + HuggingFace Transformers |

---

## Why BlenderBot?

BlenderBot was pre-trained on four datasets: 1.5B Reddit threads, ConvAI2 (PersonaChat), EmpatheticDialogues, and Wizard of Wikipedia. Unlike DialoGPT, it uses an encoder-decoder architecture that explicitly separates reading context from generating a response — this is why it handles multi-turn conversation much better.

We fine-tune on `blended_skill_talk` because it is one of BlenderBot's own training sets, giving the best domain alignment.

---

## Training Results

| Epoch | Train Loss | Train PPL | Val Loss | Val PPL |
|---|---|---|---|---|
| 1 | 2.37 | 10.72 | 2.66 | **14.23** ← best |
| 2 | 1.96 | 7.11 | 2.77 | 15.90 |
| 3 | 1.58 | 4.83 | 2.95 | 19.12 |

Best checkpoint saved at epoch 1 (early stopping based on val loss).

**Metric used: Perplexity, not BLEU.** BLEU is misleading for chatbots because "How are you?" has hundreds of valid replies — BLEU would score most of them zero against a single reference.

---

## Project Structure

```
blenderbot-chatbot/
  blenderbot_final.ipynb   Training notebook (run on Kaggle GPU)
  main.py                  FastAPI server
  index.html               Chat UI (served by FastAPI)
  requirements.txt         Python dependencies
  README.md                This file
  blenderbot_finetuned/    Trained model weights (download from Kaggle output)
    config.json
    pytorch_model.bin
    tokenizer_config.json
    vocab.json
    merges.txt
    generation_config.json
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/blenderbot-chatbot.git
cd blenderbot-chatbot

# 2. Unzip the model (downloaded from Kaggle output tab)
unzip blenderbot_finetuned.zip

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000

# 5. Open the chat UI
# Go to http://localhost:8000 in your browser
```

---

## API Usage

**Single-turn:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! How are you?"}'
```

**Multi-turn (pass history):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What skills do I need?",
    "history": ["I want to work in AI.", "That is a great goal!"]
  }'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

---

## Deploy on HuggingFace Spaces (Free, Public URL)

1. Create a new Space at https://huggingface.co/spaces
2. Select **FastAPI** as the SDK
3. Upload all files from this repo
4. Upload the `blenderbot_finetuned/` folder (or push via `huggingface_hub`)
5. HuggingFace Spaces gives you a free public URL with GPU option

---

## Deploy on Railway / Render

```bash
# Railway
railway init
railway up

# Render: connect GitHub repo, set start command:
# uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Overfitting Fixes Applied

The model showed clear overfitting (train PPL dropped fast while val PPL rose). Fixes:

- **LR reduced** from 5e-5 to 3e-5 — smaller updates, less memorisation
- **Epochs reduced** from 4 to 3 — stop training before val loss climbs
- **Samples reduced** from 40K to 20K — smaller set is harder to memorise
- **Label smoothing 0.1** — prevents model from being overconfident on training targets
- **Warmup increased** from 6% to 10% — more stable start with pretrained weights

---

## Author

Nilotpal — CS student, AI/ML enthusiast
