#!/usr/bin/env python3
"""
Vera — magicpin Merchant AI Assistant Bot
==========================================
Full FastAPI HTTP server for the magicpin AI Challenge.

Endpoints:
  GET  /v1/healthz    — liveness probe
  GET  /v1/metadata   — bot identity
  POST /v1/context    — receive context push (category/merchant/customer/trigger)
  POST /v1/tick       — periodic wake-up; bot decides what to send proactively
  POST /v1/reply      — receive merchant/customer reply; bot responds

Usage:
  pip install fastapi uvicorn google-generativeai
  export GEMINI_API_KEY=your_key_here
  uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import os
import re
import time
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from google import genai
from google.genai import types as genai_types
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEAM_NAME = "VeraPlus"
TEAM_MEMBERS = ["Sanjay"]
CONTACT_EMAIL = "team@example.com"
BOT_VERSION = "1.0.0"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vera")

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

if GEMINI_API_KEY:
    _client = genai.Client(api_key=GEMINI_API_KEY)
else:
    _client = None
    log.warning("GEMINI_API_KEY not set — LLM calls will return stubs")


def llm_complete(prompt: str) -> str:
    """Call Gemini and return the text response."""
    if _client is None:
        return json.dumps({
            "body": "Hi! Checking in — anything I can help with today? Reply YES to proceed.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "rationale": "Stub response — GEMINI_API_KEY not set",
        })
    try:
        resp = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=800,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return resp.text
    except Exception as e:
        log.error(f"Gemini error: {e}")
        raise


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

START_TIME = time.time()

# (scope, context_id) -> {version: int, payload: dict}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id -> list of turns [{from, body, ts}]
conversations: dict[str, list] = {}

# suppression_key -> True (already sent this cycle)
sent_suppression: set[str] = set()

# conversation_id -> auto_reply_count
auto_reply_counts: dict[str, int] = {}

# conversation_id -> ended (bool)
ended_conversations: set[str] = set()

# conversation_id -> unanswered_nudge_count
unanswered_nudges: dict[str, int] = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Vera — Merchant AI Assistant", version=BOT_VERSION)


from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    uptime = int(time.time() - START_TIME)
    m, s = divmod(uptime, 60)
    h, m = divmod(m, 60)
    return f"""<!DOCTYPE html>
<html>
<head>
  <title>Vera — magicpin AI Bot</title>
  <meta charset="utf-8">
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; margin: 0; padding: 40px; }}
    h1 {{ color: #a78bfa; margin-bottom: 4px; }}
    .sub {{ color: #64748b; margin-bottom: 32px; font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; max-width: 700px; }}
    .card {{ background: #1e2130; border-radius: 12px; padding: 20px 24px; }}
    .card h3 {{ margin: 0 0 8px; color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
    .card .val {{ font-size: 28px; font-weight: 700; color: #a78bfa; }}
    .card .val.ok {{ color: #34d399; }}
    .endpoints {{ margin-top: 32px; max-width: 700px; }}
    .endpoints h2 {{ color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
    a {{ color: #818cf8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .ep {{ background: #1e2130; border-radius: 8px; padding: 10px 16px; margin: 8px 0; font-family: monospace; font-size: 14px; }}
    .method {{ color: #fbbf24; margin-right: 12px; }}
  </style>
</head>
<body>
  <h1>🤖 Vera — Merchant AI Assistant</h1>
  <div class="sub">magicpin AI Challenge · Team: {TEAM_NAME} · Model: {GEMINI_MODEL}</div>
  <div class="grid">
    <div class="card">
      <h3>Status</h3>
      <div class="val ok">● ONLINE</div>
    </div>
    <div class="card">
      <h3>Uptime</h3>
      <div class="val">{h:02d}:{m:02d}:{s:02d}</div>
    </div>
    <div class="card">
      <h3>Contexts Loaded</h3>
      <div class="val">{sum(counts.values())}</div>
    </div>
    <div class="card">
      <h3>Breakdown</h3>
      <div style="font-size:13px;margin-top:4px;color:#94a3b8;">
        📁 {counts['category']} categories &nbsp;·&nbsp; 🏪 {counts['merchant']} merchants<br>
        👥 {counts['customer']} customers &nbsp;·&nbsp; ⚡ {counts['trigger']} triggers
      </div>
    </div>
  </div>
  <div class="endpoints">
    <h2>Endpoints</h2>
    <div class="ep"><span class="method">GET</span> <a href="/v1/healthz">/v1/healthz</a> — liveness probe</div>
    <div class="ep"><span class="method">GET</span> <a href="/v1/metadata">/v1/metadata</a> — bot identity</div>
    <div class="ep"><span class="method">POST</span> /v1/context — push category / merchant / customer / trigger</div>
    <div class="ep"><span class="method">POST</span> /v1/tick — wake-up; bot composes proactive messages</div>
    <div class="ep"><span class="method">POST</span> /v1/reply — receive merchant reply; bot responds</div>
    <div class="ep"><span class="method">GET</span> <a href="/docs">/docs</a> — interactive API docs (Swagger)</div>
  </div>
</body>
</html>"""



# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": GEMINI_MODEL,
        "approach": (
            "Trigger-aware prompt router -> Gemini 2.0 Flash composer. "
            "Auto-reply detection, intent-transition routing, graceful exit. "
            "Hindi-English code-mix, service+price specificity, compulsion levers."
        ),
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": "2026-09-27T00:00:00Z",
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": f"Unknown scope: {body.scope}"}

    key = (body.scope, body.context_id)
    current = contexts.get(key)
    if current and current["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": current["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    log.info(f"Context stored: {body.scope}/{body.context_id} v{body.version}")
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        trg_ctx = contexts.get(("trigger", trg_id))
        if not trg_ctx:
            continue
        trg = trg_ctx["payload"]

        # Suppression check
        sup_key = trg.get("suppression_key", trg_id)
        if sup_key in sent_suppression:
            log.info(f"Suppressed trigger {trg_id} (key={sup_key})")
            continue

        # Check expiry
        expires_at = trg.get("expires_at", "")
        if expires_at:
            try:
                exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                now_dt = datetime.fromisoformat(body.now.replace("Z", "+00:00"))
                if now_dt > exp:
                    log.info(f"Trigger {trg_id} expired")
                    continue
            except Exception:
                pass

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")

        merchant_ctx = contexts.get(("merchant", merchant_id), {}).get("payload") if merchant_id else None
        if not merchant_ctx:
            continue

        cat_slug = merchant_ctx.get("category_slug", "")
        category_ctx = contexts.get(("category", cat_slug), {}).get("payload")
        if not category_ctx:
            continue

        customer_ctx = None
        if customer_id:
            customer_ctx = contexts.get(("customer", customer_id), {}).get("payload")

        # Compose message
        try:
            composed = compose(category_ctx, merchant_ctx, trg, customer_ctx)
        except Exception as e:
            log.error(f"Compose error for {trg_id}: {e}")
            continue

        if not composed or not composed.get("body"):
            continue

        conv_id = f"conv_{merchant_id}_{trg_id}"
        send_as = composed.get("send_as", "vera")

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
            "template_params": _extract_template_params(composed["body"], merchant_ctx),
            "body": composed["body"],
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": sup_key,
            "rationale": composed.get("rationale", ""),
        }

        # Record in conversation history
        conversations.setdefault(conv_id, []).append({
            "from": "vera",
            "body": composed["body"],
            "ts": body.now,
        })

        sent_suppression.add(sup_key)
        actions.append(action)

        # Cap at 20 actions per tick
        if len(actions) >= 20:
            break

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    merchant_id = body.merchant_id
    message = body.message.strip()

    # If conversation already ended, just ack
    if conv_id in ended_conversations:
        return {"action": "end", "rationale": "Conversation already closed"}

    # Record incoming message
    conversations.setdefault(conv_id, []).append({
        "from": body.from_role,
        "body": message,
        "ts": body.received_at,
    })

    history = conversations[conv_id]

    # --- Auto-reply detection ---
    auto_reply_count = auto_reply_counts.get(conv_id, 0)

    if _is_auto_reply(message):
        auto_reply_count += 1
        auto_reply_counts[conv_id] = auto_reply_count
        if auto_reply_count >= 2:
            log.info(f"Auto-reply detected x{auto_reply_count} in {conv_id} — ending")
            ended_conversations.add(conv_id)
            return {
                "action": "end",
                "rationale": f"Detected auto-reply pattern after {auto_reply_count} occurrences; gracefully exiting.",
            }
        else:
            # One attempt to cut through auto-reply
            reengagement = _craft_reengagement(conv_id, merchant_id)
            return {
                "action": "send",
                "body": reengagement,
                "cta": "binary_yes_stop",
                "rationale": "Detected possible auto-reply; attempting one clarification before exiting.",
            }
    else:
        auto_reply_counts[conv_id] = 0  # reset on real reply

    # --- Hostility / opt-out detection ---
    if _is_hostile_or_optout(message):
        ended_conversations.add(conv_id)
        farewell = _craft_farewell(merchant_id)
        return {
            "action": "send",
            "body": farewell,
            "cta": "none",
            "rationale": "Merchant signalled disinterest/hostility; sending polite farewell and ending.",
        }

    # --- Intent-transition detection ---
    if _is_strong_commitment(message):
        unanswered_nudges[conv_id] = 0
        action_reply = _craft_action_reply(conv_id, merchant_id, message)
        return {
            "action": "send",
            "body": action_reply,
            "cta": "open_ended",
            "rationale": "Merchant committed to action; switching immediately to action mode.",
        }

    # --- Check unanswered nudge limit ---
    if _is_non_committal(message):
        unanswered_nudges[conv_id] = unanswered_nudges.get(conv_id, 0) + 1
        if unanswered_nudges.get(conv_id, 0) >= 3:
            ended_conversations.add(conv_id)
            return {
                "action": "end",
                "rationale": "Three unanswered nudges — exiting gracefully to avoid spam.",
            }
    else:
        unanswered_nudges[conv_id] = 0

    # --- Regular reply composition ---
    merchant_ctx = contexts.get(("merchant", merchant_id), {}).get("payload") if merchant_id else None
    cat_slug = merchant_ctx.get("category_slug", "") if merchant_ctx else ""
    category_ctx = contexts.get(("category", cat_slug), {}).get("payload") if cat_slug else None

    try:
        response_body = _compose_reply(
            conv_id=conv_id,
            merchant_id=merchant_id,
            merchant_message=message,
            history=history,
            merchant_ctx=merchant_ctx,
            category_ctx=category_ctx,
        )
    except Exception as e:
        log.error(f"Reply compose error: {e}")
        response_body = "Got it! Kya main aur kuch help kar sakti hoon? Reply YES to continue."

    conversations[conv_id].append({"from": "vera", "body": response_body, "ts": body.received_at})

    return {
        "action": "send",
        "body": response_body,
        "cta": "open_ended",
        "rationale": "Responding to merchant message with contextual follow-up.",
    }


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppression.clear()
    auto_reply_counts.clear()
    ended_conversations.clear()
    unanswered_nudges.clear()
    log.info("State wiped on teardown")
    return {"wiped": True}


# ---------------------------------------------------------------------------
# Core composer
# ---------------------------------------------------------------------------

TRIGGER_ROUTING = {
    "research_digest": "research",
    "category_research_digest_release": "research",
    "perf_dip": "perf_dip",
    "perf_spike": "perf_spike",
    "milestone_reached": "milestone",
    "dormant_with_vera": "dormant",
    "review_theme_emerged": "review_theme",
    "competitor_opened": "competitor",
    "festival_upcoming": "festival",
    "weather_heatwave": "weather_event",
    "local_news_event": "local_event",
    "regulation_change": "regulation",
    "category_trend_movement": "trend",
    "recall_due": "recall",
    "customer_lapsed_soft": "lapsed_customer",
    "appointment_tomorrow": "appointment",
    "chronic_refill_due": "refill",
    "trial_followup": "trial_followup",
    "renewal_due": "renewal",
    "curious_ask_due": "curious_ask",
    "scheduled_recurring": "curious_ask",
}


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """Main composition function — builds a WhatsApp message from 4 contexts."""
    trigger_kind = trigger.get("kind", "generic")
    route = TRIGGER_ROUTING.get(trigger_kind, "generic")

    scope = trigger.get("scope", "merchant")
    send_as = "merchant_on_behalf" if (scope == "customer" and customer) else "vera"

    prompt = _build_compose_prompt(category, merchant, trigger, customer, route, send_as)

    raw = llm_complete(prompt)
    parsed = _parse_composed(raw)

    parsed["send_as"] = send_as
    return parsed


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PERSONA = """You are Vera, magicpin's merchant AI assistant on WhatsApp.

HARD RULES:
1. Voice = peer/colleague, NOT promotional. Never "Amazing deal!" or "100% guaranteed".
2. Specificity wins: anchor on at least one verifiable fact (number, date, stat, citation).
3. Body must be concise for WhatsApp — 50-130 words ideal.
4. Single CTA — binary YES/STOP for action triggers; no CTA for pure-info triggers.
5. Hindi-English code-mix is natural and preferred when merchant language includes "hi".
6. Service+price format ("Cleaning @ Rs.299") beats generic "10% off".
7. DO NOT fabricate data not present in the context. If you don't have a number, don't invent one.
8. No long preamble ("I hope you're doing well..."). Get to the point.
9. Do NOT re-introduce yourself after turn 1.
10. CTA must be the LAST sentence.

COMPULSION LEVERS (use at least one):
- Loss aversion: "aap X miss kar rahe hain"
- Specificity: concrete number/date/citation
- Social proof: "X merchants in your area did Y this month"
- Effort externalization: "maine draft kar diya — just say go"
- Curiosity: "want to see who?"
- Binary commitment: "Reply YES / STOP"
"""


def _build_compose_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
    route: str,
    send_as: str,
) -> str:
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    signals = merchant.get("signals", [])
    offers = merchant.get("offers", [])
    active_offers = [o["title"] for o in offers if o.get("status") == "active"]
    peer = category.get("peer_stats", {})
    digest = category.get("digest", [])
    voice = category.get("voice", {})
    trends = category.get("trend_signals", [])
    seasonal = category.get("seasonal_beats", [])
    offer_catalog = category.get("offer_catalog", [])
    conv_hist = merchant.get("conversation_history", [])

    merchant_name = identity.get("name", "Merchant")
    owner_name = identity.get("owner_first_name", "")
    salutation = owner_name or merchant_name
    locality = identity.get("locality", "")
    city = identity.get("city", "")
    languages = identity.get("languages", ["en"])
    use_hindi = "hi" in languages

    lang_note = (
        "Use natural Hindi-English code-mix (Hinglish). "
        "Keep English for numbers/technical terms; Hindi for conversational parts."
        if use_hindi else
        "Use English only."
    )

    trigger_payload = json.dumps(trigger.get("payload", {}), ensure_ascii=False)
    trigger_kind = trigger.get("kind", "generic")
    urgency = trigger.get("urgency", 2)

    # Pick most relevant digest item
    top_digest = ""
    if digest:
        top_digest = json.dumps(digest[0], ensure_ascii=False)

    # Peer comparison
    merchant_ctr = perf.get("ctr", 0)
    peer_ctr = peer.get("avg_ctr", 0)
    ctr_gap = ""
    if merchant_ctr and peer_ctr:
        if merchant_ctr < peer_ctr:
            ctr_gap = f"Merchant CTR {merchant_ctr:.3f} is BELOW peer median {peer_ctr:.3f}."
        else:
            ctr_gap = f"Merchant CTR {merchant_ctr:.3f} is ABOVE peer median {peer_ctr:.3f}."

    # Customer info
    customer_section = ""
    if customer:
        cid = customer.get("identity", {})
        crel = customer.get("relationship", {})
        cstate = customer.get("state", "")
        cpref = customer.get("preferences", {})
        customer_section = f"""
CUSTOMER (message sent ON BEHALF of merchant to their customer):
  Name: {cid.get('name', 'Customer')}
  Language: {cid.get('language_pref', 'en')}
  State: {cstate}
  Last visit: {crel.get('last_visit', '?')}
  Visits total: {crel.get('visits_total', '?')}
  Services received: {crel.get('services_received', [])}
  Preferred slot: {cpref.get('preferred_slots', '?')}
  send_as = merchant_on_behalf (message appears from merchant's WA number)
"""

    route_instruction = _route_instruction(route, trigger, merchant, category, customer)

    prompt = f"""{SYSTEM_PERSONA}

=== COMPOSE TASK ===
Route type: {route}
Trigger kind: {trigger_kind} (urgency={urgency})
Trigger payload: {trigger_payload}

=== CATEGORY ===
Slug: {category.get('slug', '?')}
Voice tone: {voice.get('tone', '?')}
Taboos (NEVER use): {voice.get('vocab_taboo', [])}
Peer stats: avg_ctr={peer.get('avg_ctr')}, avg_reviews={peer.get('avg_review_count')}, avg_rating={peer.get('avg_rating')}
Top digest item: {top_digest}
Offer catalog (canonical service+price): {json.dumps([o['title'] for o in offer_catalog[:4]], ensure_ascii=False)}
Trend signals: {json.dumps(trends[:2], ensure_ascii=False)}
Seasonal beats: {json.dumps(seasonal[:2], ensure_ascii=False)}

=== MERCHANT ===
Name: {merchant_name}
Owner: {salutation}
Location: {locality}, {city}
Languages: {languages}
Subscription: {json.dumps(merchant.get('subscription', {}), ensure_ascii=False)}
Performance (30d): views={perf.get('views')}, calls={perf.get('calls')}, ctr={perf.get('ctr')}, directions={perf.get('directions')}
7-day delta: {json.dumps(perf.get('delta_7d', {}), ensure_ascii=False)}
{ctr_gap}
Active offers: {active_offers}
Signals: {signals}
Customer aggregate: {json.dumps(merchant.get('customer_aggregate', {}), ensure_ascii=False)}
Review themes: {merchant.get('review_themes', [])}
Recent Vera conversation (last 3 turns): {json.dumps(conv_hist[-3:], ensure_ascii=False) if conv_hist else 'None'}
{customer_section}
=== LANGUAGE ===
{lang_note}

=== SPECIFIC INSTRUCTIONS FOR THIS TRIGGER TYPE ===
{route_instruction}

=== OUTPUT FORMAT (strict JSON, no markdown fences) ===
{{
  "body": "<the WhatsApp message body>",
  "cta": "<binary_yes_stop | open_ended | none>",
  "suppression_key": "<unique dedup key>",
  "rationale": "<1-2 sentences: why this message, what compulsion lever used>"
}}

Compose now:"""

    return prompt


def _route_instruction(route: str, trigger: dict, merchant: dict, category: dict, customer: Optional[dict]) -> str:
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})
    digest = category.get("digest", [])
    signals = merchant.get("signals", [])

    if route == "research":
        item_id = payload.get("top_item_id") or payload.get("top_item", {}).get("id", "")
        digest_item = next((d for d in digest if d.get("id") == item_id), digest[0] if digest else {})
        return (
            f"Share this research finding: {json.dumps(digest_item, ensure_ascii=False)}. "
            "Reference specific numbers (trial size, % improvement, source/page). "
            "Frame as: 'relevant to YOUR patient/customer cohort'. "
            "CTA: ask if they want the full abstract or a patient-ed message they can reshare. "
            "cta=open_ended."
        )

    if route == "perf_dip":
        views = perf.get("views", "?")
        delta = perf.get("delta_7d", {}).get("views_pct", 0)
        peer_avg = peer.get("avg_views_30d", "?")
        delta_pct = f"{delta:+.0%}" if isinstance(delta, (int, float)) else str(delta)
        return (
            f"Performance dropped. Views={views} (7d delta={delta_pct}), peer avg={peer_avg}. "
            "Frame as loss aversion: 'aap X views miss kar rahe hain'. "
            "Offer one concrete fix (e.g., add a Google Post, refresh photos). "
            "cta=binary_yes_stop."
        )

    if route == "perf_spike":
        views = perf.get("views", "?")
        delta = perf.get("delta_7d", {}).get("views_pct", 0)
        delta_pct = f"{delta:+.0%}" if isinstance(delta, (int, float)) else str(delta)
        return (
            f"Performance spiked! Views={views} (7d delta={delta_pct}). "
            "Frame as: 'great momentum — let's convert it'. "
            "Suggest an active offer to capture the surge. "
            "cta=binary_yes_stop."
        )

    if route == "milestone":
        return (
            "Merchant hit a milestone (reviews, views, etc.). "
            "Celebrate briefly, then pivot to 'what's next' — suggest a higher-level goal. "
            "cta=open_ended."
        )

    if route == "dormant":
        last_signals = signals[:2] if signals else []
        return (
            f"Merchant hasn't replied to Vera in 14+ days. Signals on file: {last_signals}. "
            "Open with a curiosity hook — don't re-pitch, ask a question. "
            "Keep it very short. cta=binary_yes_stop."
        )

    if route == "review_theme":
        theme = payload.get("metric_or_topic", "service quality")
        return (
            f"A review theme emerged: '{theme}'. "
            "Reference the theme, frame it as actionable insight, not a problem. "
            "cta=open_ended."
        )

    if route == "competitor":
        return (
            "A new competitor opened nearby. "
            "Frame as social proof / urgency: 'competition is rising — let's make sure your profile stands out'. "
            "Suggest one specific differentiator action. cta=binary_yes_stop."
        )

    if route == "festival":
        festival_name = payload.get("festival") or payload.get("metric_or_topic", "upcoming festival")
        return (
            f"Festival: {festival_name}. "
            "Suggest a festival-specific offer in service+price format. "
            "Urgency: 'limited window'. cta=binary_yes_stop."
        )

    if route in ("weather_event", "local_event"):
        return (
            "Hyperlocal event (weather/news). "
            "Connect it to the merchant's business with a concrete angle. "
            "cta=open_ended."
        )

    if route == "regulation":
        return (
            "Regulatory change announced. "
            "Explain the specific change + deadline + action the merchant must take. "
            "Clinical/peer tone. cta=binary_yes_stop."
        )

    if route == "trend":
        trend_signals = category.get("trend_signals", [{}])
        trend = trend_signals[0] if trend_signals else {}
        return (
            f"Search trend: {json.dumps(trend, ensure_ascii=False)}. "
            "Frame as opportunity: 'X searches in your locality, but your profile isn't capitalizing'. "
            "Suggest specific GBP/offer action. cta=binary_yes_stop."
        )

    if route == "recall":
        if customer:
            cid = customer.get("identity", {})
            crel = customer.get("relationship", {})
            cpref = customer.get("preferences", {})
            return (
                f"Customer recall message (send_as=merchant_on_behalf). "
                f"Customer: {cid.get('name')}, last visit: {crel.get('last_visit')}, "
                f"state: {customer.get('state')}, language: {cid.get('language_pref')}. "
                f"Preferred slot: {cpref.get('preferred_slots', 'flexible')}. "
                "Offer 2 specific appointment slots. Use active offer price. "
                "Match language preference. cta=binary slot choice."
            )
        return "Patient recall due. Compose a gentle reminder for the merchant to reach their patient. cta=binary_yes_stop."

    if route == "lapsed_customer":
        if customer:
            cid = customer.get("identity", {})
            return (
                f"Customer {cid.get('name')} has lapsed. "
                "Winback message on behalf of merchant. "
                "Offer an incentive from active catalog. cta=binary_yes_stop."
            )
        return "Customer lapsed. Remind merchant to reach out. cta=binary_yes_stop."

    if route == "appointment":
        return (
            "Appointment is tomorrow. Confirmation/reminder message to customer. "
            "Short, warm, include time. cta=none (just confirmation)."
        )

    if route == "refill":
        return (
            "Chronic medication refill due for customer. "
            "Short, precise, include medicine name if in payload. cta=binary_yes_stop."
        )

    if route == "trial_followup":
        return (
            "Follow up after trial. Ask about experience. "
            "Soft pitch to convert to paid. cta=binary_yes_stop."
        )

    if route == "renewal":
        sub = merchant.get("subscription", {})
        days = sub.get("days_remaining", "?")
        return (
            f"Subscription renewal due in {days} days. "
            "Frame as 'don't lose momentum'. List 1-2 concrete things subscription enables. "
            "cta=binary_yes_stop."
        )

    if route == "curious_ask":
        return (
            "Curiosity-driven proactive question. "
            "Ask something genuinely interesting about their business this week "
            "(e.g., 'what treatment is most asked about this week?'). "
            "No pitch. Just a question to open a conversation. cta=open_ended."
        )

    # Generic fallback
    return (
        "Generic proactive message. Use the most actionable signal available. "
        "Be specific, peer tone. cta=open_ended."
    )


# ---------------------------------------------------------------------------
# Reply composer
# ---------------------------------------------------------------------------

def _compose_reply(
    conv_id: str,
    merchant_id: Optional[str],
    merchant_message: str,
    history: list,
    merchant_ctx: Optional[dict],
    category_ctx: Optional[dict],
) -> str:
    merchant_name = "Merchant"
    languages = ["en"]
    active_offers = []
    peer_stats = {}
    voice = {}

    if merchant_ctx:
        identity = merchant_ctx.get("identity", {})
        merchant_name = identity.get("name", "Merchant")
        languages = identity.get("languages", ["en"])
        offers = merchant_ctx.get("offers", [])
        active_offers = [o["title"] for o in offers if o.get("status") == "active"]
    if category_ctx:
        peer_stats = category_ctx.get("peer_stats", {})
        voice = category_ctx.get("voice", {})

    use_hindi = "hi" in languages
    lang_note = "Hinglish (Hindi-English mix preferred)" if use_hindi else "English"

    hist_str = "\n".join(
        f"[{t['from'].upper()}]: {t['body']}" for t in history[-6:]
    )

    prompt = f"""{SYSTEM_PERSONA}

=== REPLY TASK ===
You are Vera, continuing an existing conversation with {merchant_name}.
Language: {lang_note}
Active offers: {active_offers}
Peer stats: {json.dumps(peer_stats, ensure_ascii=False)}
Voice taboos: {voice.get('vocab_taboo', [])}

=== CONVERSATION HISTORY (last 6 turns) ===
{hist_str}

=== MERCHANT'S LATEST MESSAGE ===
"{merchant_message}"

=== YOUR TASK ===
Write the next Vera message. Rules:
- DO NOT re-introduce yourself
- If merchant accepted/committed -> move to action immediately (no more qualifying questions)
- If merchant asked a question -> answer it specifically
- If merchant wants to stop -> say a brief, polite goodbye
- Keep it under 100 words
- End with a clear next step if applicable

Reply with only the WhatsApp message body text (no JSON, no labels):"""

    return llm_complete(prompt)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

AUTO_REPLY_PATTERNS = [
    r"thank you for (contacting|reaching out|messaging)",
    r"our team will (respond|reply|get back)",
    r"automated (message|response|assistant|reply)",
    r"i (am|'m) (an )?automated",
    r"this is an? (auto|automated|automatic)",
    r"we will get back to you",
    r"aapki jaankari ke liye bahut.shukriya",
    r"main.*automated assistant",
    r"shukriya.*team tak pahuncha",
]

HOSTILE_PATTERNS = [
    r"\bstop\b", r"\bunsubscribe\b", r"\bblock\b", r"\bspam\b",
    r"not interested", r"do not (message|contact|call)",
    r"band karo", r"mat karo", r"nahin chahiye", r"nahi chahiye",
    r"remove.*list", r"opt.?out",
]

COMMITMENT_PATTERNS = [
    r"\byes\b", r"\bhaan\b", r"\bok(ay)?\b", r"\bgo ahead\b",
    r"\blet'?s do it\b", r"\bproceed\b", r"\bsure\b", r"\bconfirm\b",
    r"\bkar do\b", r"\bkaro\b", r"\bkijiye\b", r"\bchalo\b",
    r"\bstart\b", r"\bbegin\b", r"\bgo\b",
]

NON_COMMITTAL_PATTERNS = [
    r"\bmaybe\b", r"\bbaad mein\b", r"\blater\b", r"\bthink about it\b",
    r"\bnot now\b", r"\babhi nahi\b", r"\bdekhte hain\b",
    r"\bsoch.*hoon\b", r"\bbusy\b",
]


def _is_auto_reply(message: str) -> bool:
    msg_lower = message.lower()
    for pat in AUTO_REPLY_PATTERNS:
        if re.search(pat, msg_lower):
            return True
    return False


def _is_hostile_or_optout(message: str) -> bool:
    msg_lower = message.lower()
    for pat in HOSTILE_PATTERNS:
        if re.search(pat, msg_lower):
            return True
    return False


def _is_strong_commitment(message: str) -> bool:
    msg_lower = message.lower()
    for pat in COMMITMENT_PATTERNS:
        if re.search(pat, msg_lower):
            return True
    return False


def _is_non_committal(message: str) -> bool:
    msg_lower = message.lower()
    for pat in NON_COMMITTAL_PATTERNS:
        if re.search(pat, msg_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Craft helpers (fast, no LLM call)
# ---------------------------------------------------------------------------

def _craft_reengagement(conv_id: str, merchant_id: Optional[str]) -> str:
    merchant_ctx = contexts.get(("merchant", merchant_id), {}).get("payload") if merchant_id else {}
    name = merchant_ctx.get("identity", {}).get("owner_first_name", "") or "aap" if merchant_ctx else "aap"
    return (
        f"Hi {name}! Yeh message directly aap ke liye hai — ek quick question: "
        "kya main aapke Google profile ka ek specific update kar doon? "
        "2 minute ka kaam hai. Reply YES / STOP."
    )


def _craft_farewell(merchant_id: Optional[str]) -> str:
    merchant_ctx = contexts.get(("merchant", merchant_id), {}).get("payload") if merchant_id else {}
    name = merchant_ctx.get("identity", {}).get("name", "aapka business") if merchant_ctx else "aapka business"
    return f"Bilkul samajh gayi. {name} ke liye best wishes! Jab bhi zaroorat ho, hum yahaan hain."


def _craft_action_reply(conv_id: str, merchant_id: Optional[str], message: str) -> str:
    merchant_ctx = contexts.get(("merchant", merchant_id), {}).get("payload") if merchant_id else {}
    identity = merchant_ctx.get("identity", {}) if merchant_ctx else {}
    offers = merchant_ctx.get("offers", []) if merchant_ctx else []
    active_offers = [o["title"] for o in offers if o.get("status") == "active"]

    offer_line = f"Active offer: {active_offers[0]}. " if active_offers else ""

    return (
        f"Done! {offer_line}"
        "Main abhi aapke Google profile pe kaam shuru kar rahi hoon. "
        "Updates mil jayenge 24-48 ghante mein. Kya kuch aur bhi add karna hai?"
    )


def _extract_template_params(body: str, merchant: dict) -> list:
    """Extract up to 3 template parameters from the composed body."""
    name = merchant.get("identity", {}).get("name", "Merchant")
    sentences = re.split(r'(?<=[.!?])\s+', body.strip())
    params = [name] + sentences[:2]
    return params[:3]


# ---------------------------------------------------------------------------
# JSON parser for LLM output
# ---------------------------------------------------------------------------

def _parse_composed(raw: str) -> dict:
    """Parse JSON from LLM output, handle markdown fences."""
    text = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group())
            return {
                "body": str(data.get("body", "")).strip(),
                "cta": str(data.get("cta", "open_ended")),
                "suppression_key": str(data.get("suppression_key", str(uuid.uuid4()))),
                "rationale": str(data.get("rationale", "")),
            }
        except json.JSONDecodeError:
            pass

    log.warning("Could not parse LLM JSON; using raw text as body")
    return {
        "body": raw.strip()[:500],
        "cta": "open_ended",
        "suppression_key": str(uuid.uuid4()),
        "rationale": "Fallback — JSON parse failed",
    }
