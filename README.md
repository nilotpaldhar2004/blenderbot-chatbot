---
license: mit
language:
- en
base_model: facebook/blenderbot-400M-distill
datasets:
- blended_skill_talk
pipeline_tag: text-generation
tags:
- blenderbot
- conversational
- chatbot
- fine-tuned
- pytorch
metrics:
- perplexity
library_name: transformers
---
 
<div align="center">
 
# BlenderBot Conversational AI
 
**Fine-tuned `facebook/blenderbot-400M-distill` for open-domain multi-turn conversation**
 
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace_Space-blue?style=for-the-badge)](https://nilotpaldhar2004-blenderbot-chatbot.hf.space)
[![Model](https://img.shields.io/badge/🧠_Model-HuggingFace_Hub-orange?style=for-the-badge)](https://huggingface.co/nilotpaldhar2004/blenderbot-chatbot)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
 
</div>
---
 
## ✨ Live Demo
 
**Try it now:** [https://nilotpaldhar2004-blenderbot-chatbot.hf.space](https://nilotpaldhar2004-blenderbot-chatbot.hf.space)
 
> Chat with a fine-tuned BlenderBot — multi-turn conversation, empathetic responses, works on mobile and desktop.
 
---
 
## 📊 Model Details
 
| Property | Value |
|---|---|
| 🏗️ Base model | `facebook/blenderbot-400M-distill` |
| 🔢 Parameters | 364.8M |
| 📦 Dataset | `blended_skill_talk` (Facebook AI Research) |
| 🏛️ Architecture | Encoder-Decoder (seq2seq) |
| 📉 Best Val PPL | **14.16** |
| 🔧 Framework | PyTorch + HuggingFace Transformers |
| ☁️ Hosting | HuggingFace Spaces (free CPU tier) |
 
---
 
## 🧠 Why BlenderBot?
 
BlenderBot was pre-trained on **four diverse datasets**:
 
| Dataset | Size | Focus |
|---|---|---|
| Reddit comments | 1.5B threads | General conversation |
| ConvAI2 (PersonaChat) | 160K turns | Persona-grounded dialogue |
| EmpatheticDialogues | 25K conversations | Emotional awareness |
| Wizard of Wikipedia | 22K conversations | Knowledge-grounded chat |
 
Unlike DialoGPT (decoder-only), BlenderBot uses an **encoder-decoder architecture** that separately reads conversation history and generates responses — giving it significantly better multi-turn coherence.
 
Fine-tuned on `blended_skill_talk` because it is one of BlenderBot's own training sets, giving the best possible domain alignment.
 
---
 
## 📈 Training Results
 
| Epoch | Train Loss | Train PPL | Val Loss | Val PPL |
|---|---|---|---|---|
| 1 | 2.37 | 10.68 | 2.62 | **14.16 ✅ best** |
| 2 | 2.05 | 7.75 | 2.67 | 14.44 |
 
> ⏹️ Early stopping triggered at epoch 2. Best checkpoint saved at epoch 1.
 
### Why Perplexity, not BLEU?
 
BLEU compares output to a single reference answer. But *"How are you?"* has hundreds of valid replies — BLEU would score most of them zero. **Perplexity measures how confident the model is on held-out data**, making it the correct metric for open-domain chatbots.
 
---
 
## 💬 Sample Conversations
 
```
User : Hello! How are you doing today?
Bot  : I'm doing well, how about yourself?
 
User : I am feeling really stressed about work.
Bot  : I'm sorry to hear that. What's going on?
 
User : I just got promoted at my job!
Bot  : Congratulations! What do you do for a living?
 
User : Do you have any advice for staying healthy?
Bot  : I try to eat as healthy as I can and stay active.
 
User : What do you think about artificial intelligence?
Bot  : I think it's a great idea. It could be useful in many ways.
```
 
---
 
## 🚀 Quick Start
 
```bash
# 1. Clone the repo
git clone https://github.com/nilotpaldhar2004/blenderbot-chatbot.git
cd blenderbot-chatbot
 
# 2. Install dependencies
pip install -r requirements.txt
 
# 3. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
 
# 4. Open browser
# http://localhost:8000
```
 
> **Note:** Model weights (~1.5GB) are downloaded automatically from HuggingFace Hub on first run.
 
---
 
## 🔌 API Usage
 
**Health check:**
```bash
curl https://nilotpaldhar2004-blenderbot-chatbot.hf.space/health
```
 
**Single-turn:**
```bash
curl -X POST https://nilotpaldhar2004-blenderbot-chatbot.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! How are you?"}'
```
 
**Multi-turn (with history):**
```bash
curl -X POST https://nilotpaldhar2004-blenderbot-chatbot.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What skills do I need?",
    "history": ["I want to work in AI.", "That is a great goal!"]
  }'
```
 
**Python:**
```python
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
 
tokenizer = BlenderbotTokenizer.from_pretrained("nilotpaldhar2004/blenderbot-chatbot")
model     = BlenderbotForConditionalGeneration.from_pretrained("nilotpaldhar2004/blenderbot-chatbot")
 
inputs = tokenizer("Hello! How are you?", return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=60,
    num_beams=2,
    no_repeat_ngram_size=3,
    early_stopping=True,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
 
---
 
## 🗂️ Project Structure
 
```
blenderbot-chatbot/
├── main.py                  FastAPI server
├── index.html               Chat UI (responsive, mobile-ready)
├── requirements.txt         Python dependencies
├── Dockerfile               HuggingFace Spaces deployment
├── README.md                This file
├── blenderbot_final.ipynb   Training notebook (Kaggle GPU)
└── .github/
    └── workflows/
        └── deploy.yml       Auto-deploy to HF Spaces on git push
```
 
---
 
## 🏗️ Tech Stack
 
| Layer | Technology |
|---|---|
| Model | facebook/blenderbot-400M-distill |
| Training | PyTorch + HuggingFace Transformers |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Hosting | HuggingFace Spaces (Docker) |
| Model storage | HuggingFace Model Hub |
| CI/CD | GitHub Actions -> auto-deploy to HF Spaces |
 
---
 
## 🛠️ Overfitting Fixes Applied
 
| Fix | Before | After |
|---|---|---|
| Learning rate | 5e-5 | 2e-5 |
| Epochs | 4 | 2 (early stop) |
| Early stopping patience | none | 2 |
| Warmup ratio | 6% | 15% |
| Effective batch size | 32 | 64 |
 
---
 
## 👤 Author
 
<div align="center">
 
**Nilotpal Dhar** — CS student, AI/ML enthusiast
 
[![GitHub](https://img.shields.io/badge/GitHub-nilotpaldhar2004-black?style=flat&logo=github)](https://github.com/nilotpaldhar2004)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-nilotpaldhar2004-orange?style=flat)](https://huggingface.co/nilotpaldhar2004)
[![Email](https://img.shields.io/badge/Email-dharnilotpal31@gmail.com-red?style=flat&logo=gmail)](mailto:dharnilotpal31@gmail.com)
 
</div>
