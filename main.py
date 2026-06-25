"""
FinCouple AI — LINE Bot Webhook Backend
Phase 2: FastAPI + Claude AI + Supabase
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request

load_dotenv()
from fastapi.responses import JSONResponse
from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from supabase import Client, create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LINE_CHANNEL_SECRET: str = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN: str = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
CLAUDE_API_KEY: str = os.environ["CLAUDE_API_KEY"]
SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]

FINCOUPLE_SYSTEM_PROMPT = """
# Role
You are "FinCouple AI", an expert personal and couple finance assistant chatbot backend.
Your job is to process natural language text from users, extract financial data accurately,
and output it EXCLUSIVELY in a structured JSON format.

# Context & Logic
- The application allows couples to manage shared budgets split into multiple categories.
- Every transaction belongs to a specific group (couple).
- Users will either:
  1. Record an Expense
  2. Record an Income
  3. Set/Update a Budget for a category
  4. Ask for a summary or financial status

# Category Guidelines
Map the user's intent to one of these standard categories:
- food
- travel
- home
- shopping
- entertainment
- savings
- income
- other

# Output Format
You must return ONLY a valid JSON object. No prose, no markdown.

## JSON Schema:
{
  "intent": "record_expense" | "record_income" | "set_budget" | "ask_summary" | "unknown",
  "data": {
    "amount": float or null,
    "category": "food"|"travel"|"home"|"shopping"|"entertainment"|"savings"|"income"|"other"|null,
    "memo": "string description" or null,
    "target_period": "current_month"|"weekly"|"monthly" or null
  },
  "confidence": float (0.0 to 1.0),
  "error_message": "string if data is missing or ambiguous, otherwise null"
}

# Strict Rule
Never break character. Always return valid JSON only.
"""

supabase: Client | None = None
anthropic_client: anthropic.AsyncAnthropic | None = None
line_parser: WebhookParser | None = None
line_config: Configuration | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase, anthropic_client, line_parser, line_config
    logger.info("Starting FinCouple AI backend...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    anthropic_client = anthropic.AsyncAnthropic(api_key=CLAUDE_API_KEY)
    line_parser = WebhookParser(LINE_CHANNEL_SECRET)
    line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
    logger.info("All clients initialised.")
    yield
    logger.info("Shutting down FinCouple AI backend.")


app = FastAPI(title="FinCouple AI Webhook", version="2.0.0", lifespan=lifespan)


async def get_user(line_user_id: str) -> dict[str, Any] | None:
    result = (
        supabase.table("users")
        .select("line_user_id, display_name, group_id")
        .eq("line_user_id", line_user_id)
        .maybe_single()
        .execute()
    )
    return result.data


async def upsert_user(line_user_id: str, display_name: str) -> None:
    supabase.table("users").upsert(
        {"line_user_id": line_user_id, "display_name": display_name},
        on_conflict="line_user_id",
        ignore_duplicates=True,
    ).execute()


async def insert_transaction(
    group_id: str,
    created_by: str,
    tx_type: str,
    amount: float,
    category: str,
    memo: str | None,
) -> dict[str, Any]:
    result = (
        supabase.table("transactions")
        .insert({
            "group_id": group_id,
            "created_by": created_by,
            "type": tx_type,
            "amount": amount,
            "category": category or "other",
            "memo": memo,
        })
        .execute()
    )
    return result.data[0]


async def parse_with_claude(user_message: str) -> dict[str, Any]:
    response = await anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=512,
        system=FINCOUPLE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw_text: str = response.content[0].text.strip()
    logger.info("Claude raw response: %s", raw_text)
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    return json.loads(raw_text)


CATEGORY_EMOJI: dict[str, str] = {
    "food": "🍜", "travel": "🚗", "home": "🏠", "shopping": "🛍️",
    "entertainment": "🎬", "savings": "💰", "income": "💵", "other": "📝",
}


def build_expense_reply(amount: float, category: str, memo: str | None) -> str:
    emoji = CATEGORY_EMOJI.get(category, "📝")
    memo_line = f"\n📌 {memo}" if memo else ""
    return f"{emoji} บันทึกรายจ่ายแล้ว!\n💸 ยอด: {amount:,.2f} บาท\n📂 หมวด: {category}{memo_line}"


def build_income_reply(amount: float, category: str, memo: str | None) -> str:
    memo_line = f"\n📌 {memo}" if memo else ""
    return f"💵 บันทึกรายรับแล้ว!\n✅ ยอด: {amount:,.2f} บาท\n📂 หมวด: {category}{memo_line}"


async def handle_text_event(event: MessageEvent) -> str:
    line_user_id: str = event.source.user_id
    user_text: str = event.message.text.strip()

    await upsert_user(line_user_id, display_name=line_user_id)
    user = await get_user(line_user_id)
    group_id: str | None = user.get("group_id") if user else None

    if not group_id:
        return (
            "👋 สวัสดี! ยังไม่ได้อยู่ในกลุ่มนะครับ\n"
            "พิมพ์ /create เพื่อสร้างกลุ่มใหม่\n"
            "หรือพิมพ์ /join <รหัส> เพื่อเข้าร่วมกลุ่มที่มีอยู่"
        )

    try:
        parsed = await parse_with_claude(user_text)
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error("Claude parse error: %s", exc)
        return "⚠️ AI ประมวลผลไม่ได้ในขณะนี้ กรุณาลองใหม่อีกครั้งนะครับ"

    intent: str = parsed.get("intent", "unknown")
    data: dict | None = parsed.get("data")
    error_msg: str | None = parsed.get("error_message")

    if intent in ("record_expense", "record_income"):
        if not data or data.get("amount") is None:
            return f"❓ {error_msg or 'ไม่พบยอดเงินในข้อความ กรุณาระบุจำนวนเงินด้วยครับ'}"

        amount: float = float(data["amount"])
        category: str = data.get("category") or "other"
        memo: str | None = data.get("memo")
        tx_type = "expense" if intent == "record_expense" else "income"

        try:
            await insert_transaction(
                group_id=group_id, created_by=line_user_id,
                tx_type=tx_type, amount=amount, category=category, memo=memo,
            )
        except Exception as exc:
            logger.error("Supabase insert error: %s", exc)
            return "⚠️ บันทึกข้อมูลไม่สำเร็จ กรุณาลองใหม่อีกครั้งนะครับ"

        if tx_type == "expense":
            return build_expense_reply(amount, category, memo)
        else:
            return build_income_reply(amount, category, memo)

    elif intent == "set_budget":
        return "⚙️ ฟีเจอร์ตั้งงบประมาณจะมาเร็วๆ นี้นะครับ! 🚧"

    elif intent == "ask_summary":
        return "📊 ฟีเจอร์ดูสรุปยอดจะมาเร็วๆ นี้นะครับ! 🚧"

    else:
        return f"❓ {error_msg or 'ไม่เข้าใจข้อความ กรุณาลองพิมพ์ใหม่นะครับ'}"


@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "FinCouple AI"})


@app.post("/webhook")
async def webhook(
    request: Request,
    x_line_signature: str = Header(alias="X-Line-Signature"),
) -> JSONResponse:
    body: bytes = await request.body()
    try:
        events = line_parser.parse(body.decode("utf-8"), x_line_signature)
    except InvalidSignatureError:
        logger.warning("Invalid LINE signature received.")
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessageContent):
            continue
        reply_text = await handle_text_event(event)
        with ApiClient(line_config) as api_client:
            line_api = MessagingApi(api_client)
            line_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)],
                )
            )

    return JSONResponse({"status": "ok"})
