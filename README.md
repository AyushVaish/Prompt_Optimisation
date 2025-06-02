# 🔬 Multi-LLM Prompt-Optimization Playground

*A Streamlit web-app for **designing, A/B-testing, and analysing prompts** across OpenAI GPT-4o-mini, Anthropic Claude-3 Haiku, and Mistral 7-B models.*

<div align="center">
  <img src="https://raw.githubusercontent.com/your-org/your-repo/main/.github/screenshot.png" width="700" alt="UI screenshot">
</div>

---

## 1. Why this project?

Hiring pipelines often look for engineers who can ship an end-to-end AI product—not just call an API.  
This repo shows exactly that:

* **Product thinking** – a polished UI with five task-oriented pages
* **Engineering rigour** – modular model layer + cost & latency tracking 
* **Research-grade evaluation** – BLEU, ROUGE-L and embedding-cosine metrics  
* **Data discipline** – every run is appended to `history.jsonl` for auditability  

---

## 2. Live demo pages

| Page | What it does | Core file |
|------|--------------|-----------|
| **Single Model Demo** | Send one prompt → see output, token usage, $$ and latency. | `app.py`  |
| **Compare Two Models** | Side-by-side outputs + auto prompt-improvement suggestions. | `app.py` |
| **A/B Test – Weighted-Sum** | Optimise prompts when *no ground truth* exists (quality×cost×latency WSM). | `app.py` |
| **A/B Test – Ground-Truth** | Batch compare summaries against reference text via BLEU / ROUGE / embeddings. | `app.py` |
| **Analytics Dashboard** | Aggregated spend, call counts & latency over time from `history.jsonl`. | `app.py` |

---

## 3. Architecture at a glance

```text
                ┌─────────────┐
                │  Streamlit  │  UI / charts / forms
                └─────┬───────┘
                      │                          .env → API keys
   constants.py →  Best-practice copy           ────────────────
                      │
app.py ───► models.py ─► External LLM APIs (OpenAI, Anthropic, Mistral)
                      │
      metrics.py ──► Quality scores (BLEU, ROUGE-L, cosine) 
       costs.py ───► Cost calculators per vendor 
      logger.py ───► JSONL telemetry store 
