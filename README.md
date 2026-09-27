# Vera+ — magicpin AI Challenge Submission

## Approach

**Vera+ is a trigger-aware, LLM-powered WhatsApp AI assistant** built on FastAPI + Google Gemini 2.0 Flash.

### Architecture

```
Judge → POST /v1/context      → in-memory context store (categories, merchants, customers, triggers)
Judge → POST /v1/tick         → trigger router → per-trigger prompt → Gemini → action[]
Judge → POST /v1/reply        → intent classifier → (auto-reply / hostile / commitment / regular) → Gemini reply
```

### Core Design Decisions

**1. Trigger-aware routing (not one-size-fits-all)**  
Different trigger kinds (`research_digest`, `perf_dip`, `competitor_opened`, `recall_due`, etc.) get different prompt templates with tailored instructions — framing, CTA shape, compulsion lever, and tone all vary by route type.

**2. Specificity-first prompting**  
Every compose prompt explicitly instructs Gemini to anchor on verifiable facts from the context (numbers, citations, dates). "10% off" is forbidden; "Cleaning @ ₹299" is the template.

**3. Auto-reply detection**  
Regex patterns match WhatsApp Business canned auto-replies (both English and Hindi). On first detection: one re-engagement attempt. On second consecutive auto-reply: graceful exit (`action: end`).

**4. Intent-transition without qualifying**  
If the merchant says "yes / haan / ok / let's do it / go ahead / chalo", the bot immediately switches to action mode — no additional qualifying questions. Returns `action: send` with a confirmation + next step.

**5. Graceful exit**  
- Hostile/opt-out message → polite farewell + `action: end`  
- 3+ non-committal replies → `action: end` (no spam)  
- 2+ auto-replies → `action: end`

**6. Hindi-English code-mix**  
When `merchant.identity.languages` includes `"hi"`, the prompt instructs Gemini to produce Hinglish naturally — Hindi for conversational parts, English for numbers and technical terms.

**7. Voice compliance by category**  
- Dentists: peer/clinical, `vocab_taboo` enforced (no "guaranteed", no "cure")  
- Salons: warm/practical  
- Restaurants: operator-operator  
- Gyms: coaching/motivational  
- Pharmacies: trustworthy/precise

### Compulsion levers used

| Lever | Trigger types |
|---|---|
| Loss aversion | `perf_dip`, `competitor_opened`, `dormant` |
| Specificity/verifiability | all routes (explicit in prompt) |
| Social proof | `perf_spike`, `competitor_opened`, `milestone` |
| Effort externalization | `recall_due`, `research_digest` |
| Curiosity | `curious_ask`, `review_theme` |
| Binary commitment | `renewal_due`, `perf_dip`, `festival_upcoming` |

### What would have helped most

- **Real merchant conversation history** — knowing what Vera already sent (and how the merchant replied) would dramatically improve message variety and anti-repetition.
- **Real slot availability** for appointment/recall messages — currently we say "upcoming slots" generically; actual calendar data would enable concrete slot offers like "Wed 6pm / Thu 5pm".
- **Competitor names** for `competitor_opened` triggers — the brief says "don't fabricate" so we can only say "a new competitor nearby" rather than naming them.

## Setup

```bash
pip install fastapi uvicorn google-generativeai
export GEMINI_API_KEY=your_key_here
uvicorn bot:app --host 0.0.0.0 --port 8080
```

## Generate submission.jsonl

```bash
python dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded
python generate_submission.py
```

## Run judge simulator

```bash
python judge_simulator.py   # configure BOT_URL and LLM_API_KEY in the file first
```

## Files

| File | Purpose |
|---|---|
| `bot.py` | Full FastAPI server — all 5 endpoints |
| `generate_submission.py` | Generates `submission.jsonl` from test pairs |
| `submission.jsonl` | 30 pre-composed messages for canonical test pairs |
| `dataset/expanded/` | Full expanded dataset (50 merchants, 200 customers, 100 triggers) |
