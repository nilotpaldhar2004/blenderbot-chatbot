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

# BlenderBot Conversational Chatbot

Fine-tuned `facebook/blenderbot-400M-distill` on the `blended_skill_talk` dataset for open-domain multi-turn conversation.

---

## Model

| Property | Value |
|---|---|
| Base model | `facebook/blenderbot-400M-distill` |
| Parameters | 364.8M |
| Dataset | `blended_skill_talk` (Facebook AI Research) |
| Architecture | Encoder-Decoder (seq2seq) |
| Best Val PPL | 14.16 |
| Framework | PyTorch + HuggingFace Transformers |

---

## Why BlenderBot?

BlenderBot was pre-trained on four datasets: 1.5B Reddit threads, ConvAI2 (PersonaChat), EmpatheticDialogues, and Wizard of Wikipedia. Unlike DialoGPT, it uses an encoder-decoder architecture that explicitly separates reading context from generating a response — this is why it handles multi-turn conversation much better.

Fine-tuned on `blended_skill_talk` because it is one of BlenderBot's own training sets, giving the best domain alignment.

---

## Training Results

| Epoch | Train Loss | Train PPL | Val Loss | Val PPL |
|---|---|---|---|---|
| 1 | 2.37 | 10.68 | 2.62 | **14.16** <- best |
| 2 | 2.05 | 7.75 | 2.67 | 14.44 |

Best checkpoint saved at epoch 1. Early stopping triggered at epoch 2.

**Metric: Perplexity, not BLEU.** BLEU is misleading for chatbots — "How are you?" has hundreds of valid replies and BLEU would score most of them zero against a single reference. Perplexity measures how confident the model is on held-out data.

---

## Sample Responses

```
User : Hello! How are you doing today?
Bot  : I'm doing well, how about yourself?

User : I am feeling really stressed about work.
Bot  : I'm sorry to hear that. What's going on?

User : I just got promoted at my job!
Bot  : Congratulations! What do you do for a living?

User : Do you have any advice for staying healthy?
Bot  : I try to eat as healthy as I can and stay active.
```

---

## Usage

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

## Project Structure

```
blenderbot-chatbot/
  main.py                  FastAPI server
  index.html               Chat UI
  requirements.txt         Python dependencies
  Dockerfile               HuggingFace Spaces deployment
  README.md                This file
  blenderbot_final.ipynb   Training notebook (Kaggle GPU)
  .github/workflows/
    deploy.yml             Auto-deploy to HF Spaces on git push
```

---

## Deployment

Hosted on HuggingFace Spaces (Docker SDK, free CPU tier).  
Model weights loaded from HuggingFace Model Hub at startup.

```
Space  : https://huggingface.co/spaces/nilotpaldhar2004/blenderbot-chatbot
Model  : https://huggingface.co/nilotpaldhar2004/blenderbot-chatbot
```

---

## API

```bash
# Health check
curl https://nilotpaldhar2004-blenderbot-chatbot.hf.space/health

# Single-turn
curl -X POST https://nilotpaldhar2004-blenderbot-chatbot.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Multi-turn
curl -X POST https://nilotpaldhar2004-blenderbot-chatbot.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What skills do I need?", "history": ["I want to work in AI.", "That is a great goal!"]}'
```

---

## Overfitting Fixes Applied

- LR reduced from 5e-5 to 2e-5
- Early stopping with patience=2
- Warmup increased to 15% of total steps
- Effective batch size 64 (8 x grad_accum 8)

---

## Author

**Nilotpal** — CS student, AI/ML enthusiast  
GitHub: [nilotpaldhar2004](https://github.com/nilotpaldhar2004)  
HuggingFace: [nilotpaldhar2004](https://huggingface.co/nilotpaldhar2004)
