"""
FinCouple AI — LINE Bot Webhook Backend
Phase 4: LIFF Dashboard API

Author : FinCouple Team
Python : 3.11+
"""

from __future__ import annotations

import json
import logging
import os
import random
import string
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query, Request

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------------
LINE_CHANNEL_SECRET: str = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN: str = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
CLAUDE_API_KEY: str = os.environ["CLAUDE_API_KEY"]
SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]

# ---------------------------------------------------------------------------
# System Prompt for Claude
# ---------------------------------------------------------------------------
FINCOUPLE_SYSTEM_PROMPT = """
# Role
You are "FinCouple AI", an expert personal and couple finance assistant chatbot backend.
Your job is to process natural language text from users, extract financial data accurately,
and output it EXCLUSIVELY in a structured JSON format.

# Context & Logic
- The application allows couples to manage shared budgets split into multiple categories.
- Every transaction belongs to a specific group (couple).
- Users will either:
  1. Record an Expense (รายจ่าย)
  2. Record an Income (รายรับ)
  3. Set/Update a Budget for a category (ตั้งงบประมาณ)
  4. Ask for a summary or financial status (ดูสรุป/สอบถาม)

# Category Guidelines
Map the user's intent to one of these standard categories:
- food (อาหารและเครื่องดื่ม)
- travel (เดินทาง, ค่าน้ำมัน, ค่ารถ)
- home (ที่อยู่อาศัย, ค่าน้ำ, ค่าไฟ, ค่าเน็ต, ของใช้ในบ้าน)
- shopping (ช้อปปิ้ง, เสื้อผ้า, ของฟุ่มเฟือย)
- entertainment (บันเทิง, ดูหนัง, ท่องเที่ยว, ปาร์ตี้)
- savings (เงินออม, ลงทุน)
- income (รายรับ เช่น เงินเดือน, โบนัส, ขายของ)
- other (อื่นๆ)

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

# Examples
User: "กินส้มตำกับแฟนไป 450 บาท"
Output: {"intent":"record_expense","data":{"amount":450.00,"category":"food","memo":"กินส้มตำกับแฟน","target_period":null},"confidence":0.98,"error_message":null}

User: "เงินเดือนออกแล้วจ้า 35000"
Output: {"intent":"record_income","data":{"amount":35000.00,"category":"income","memo":"เงินเดือน","target_period":null},"confidence":0.95,"error_message":null}

User: "ตั้งงบค่าอาหารเดือนนี้ 5000 บาท"
Output: {"intent":"set_budget","data":{"amount":5000.00,"category":"food","memo":"งบค่าอาหาร","target_period":"monthly"},"confidence":0.95,"error_message":null}

User: "ขอดูสรุปยอดเดือนนี้"
Output: {"intent":"ask_summary","data":{"amount":null,"category":null,"memo":null,"target_period":"current_month"},"confidence":0.90,"error_message":null}

User: "ซื้อของเข้าห้อง"
Output: {"intent":"unknown","data":null,"confidence":0.30,"error_message":"Missing transaction amount. Please provide the cost."}

# Strict Rule
Never break character. Always return valid JSON only.
"""

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
supabase: Client | None = None
anthropic_client: anthropic.AsyncAnthropic | None = None
line_parser: WebhookParser | None = None
line_config: Configuration | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase, anthropic_client, line_parser, line_config
    logger.info("🚀 Starting FinCouple AI backend…")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    anthropic_client = anthropic.AsyncAnthropic(api_key=CLAUDE_API_KEY)
    line_parser = WebhookParser(LINE_CHANNEL_SECRET)
    line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
    logger.info("✅ All clients initialised.")
    yield
    logger.info("🛑 Shutting down FinCouple AI backend.")


app = FastAPI(title="FinCouple AI Webhook", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ===========================================================================
# SUPABASE HELPERS — Users
# ===========================================================================

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


# ===========================================================================
# SUPABASE HELPERS — Groups
# ===========================================================================

def _generate_invite_code(length: int = 6) -> str:
    """Generate a random uppercase alphanumeric invite code."""
    chars = string.ascii_uppercase + string.digits
    # Remove ambiguous chars (0, O, I, 1)
    chars = chars.translate(str.maketrans("", "", "0O1I"))
    return "".join(random.choices(chars, k=length))


async def create_group(line_user_id: str) -> dict[str, Any]:
    """Create a new group, assign creator, return group row."""
    invite_code = _generate_invite_code()

    result = supabase.table("groups").insert({
        "invite_code": invite_code,
        "created_by": line_user_id,
    }).execute()

    group = result.data[0]

    # Assign user to this group
    supabase.table("users").update(
        {"group_id": group["id"]}
    ).eq("line_user_id", line_user_id).execute()

    return group


async def join_group(line_user_id: str, invite_code: str) -> dict[str, Any] | None:
    """Find group by invite code and assign user. Returns group or None."""
    result = (
        supabase.table("groups")
        .select("*")
        .eq("invite_code", invite_code.upper())
        .maybe_single()
        .execute()
    )
    if not result.data:
        return None

    group = result.data
    supabase.table("users").update(
        {"group_id": group["id"]}
    ).eq("line_user_id", line_user_id).execute()

    return group


# ===========================================================================
# SUPABASE HELPERS — Budgets
# ===========================================================================

async def upsert_budget(
    group_id: str,
    category: str,
    amount: float,
    period: str = "monthly",
) -> dict[str, Any]:
    """Upsert a budget entry (update if exists, insert if not)."""
    result = supabase.table("budgets").upsert(
        {
            "group_id": group_id,
            "category": category,
            "amount": amount,
            "period": period,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="group_id,category,period",
    ).execute()
    return result.data[0]


async def get_monthly_summary(group_id: str) -> dict[str, Any]:
    """Query current month transactions + budgets and return summary dict."""
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()

    # Transactions this month
    tx_result = (
        supabase.table("transactions")
        .select("type, amount, category, memo, created_at")
        .eq("group_id", group_id)
        .gte("created_at", month_start)
        .order("created_at", desc=True)
        .execute()
    )

    # Monthly budgets
    budget_result = (
        supabase.table("budgets")
        .select("category, amount")
        .eq("group_id", group_id)
        .eq("period", "monthly")
        .execute()
    )

    transactions = tx_result.data or []
    budgets: dict[str, float] = {
        b["category"]: float(b["amount"]) for b in (budget_result.data or [])
    }

    expense_by_cat: dict[str, float] = {}
    total_income = 0.0
    total_expense = 0.0

    for tx in transactions:
        amt = float(tx["amount"])
        if tx["type"] == "expense":
            cat = tx["category"] or "other"
            expense_by_cat[cat] = expense_by_cat.get(cat, 0.0) + amt
            total_expense += amt
        else:
            total_income += amt

    return {
        "month": now.strftime("%B %Y"),
        "total_income": total_income,
        "total_expense": total_expense,
        "balance": total_income - total_expense,
        "expense_by_category": expense_by_cat,
        "budgets": budgets,
        "recent_transactions": transactions[:20],
    }


# ===========================================================================
# CLAUDE HELPER
# ===========================================================================

async def parse_with_claude(user_message: str) -> dict[str, Any]:
    response = await anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=FINCOUPLE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw_text: str = response.content[0].text.strip()
    logger.info("Claude raw: %s", raw_text)

    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    return json.loads(raw_text)


# ===========================================================================
# REPLY BUILDERS
# ===========================================================================

CATEGORY_EMOJI: dict[str, str] = {
    "food": "🍜",
    "travel": "🚗",
    "home": "🏠",
    "shopping": "🛍️",
    "entertainment": "🎬",
    "savings": "💰",
    "income": "💵",
    "other": "📝",
}

PERIOD_TH: dict[str, str] = {
    "monthly": "รายเดือน",
    "weekly": "รายสัปดาห์",
    "current_month": "รายเดือน",
}


def build_expense_reply(amount: float, category: str, memo: str | None) -> str:
    emoji = CATEGORY_EMOJI.get(category, "📝")
    memo_line = f"\n📌 {memo}" if memo else ""
    return (
        f"{emoji} บันทึกรายจ่ายแล้ว!\n"
        f"💸 ยอด: {amount:,.2f} บาท\n"
        f"📂 หมวด: {category}{memo_line}"
    )


def build_income_reply(amount: float, category: str, memo: str | None) -> str:
    memo_line = f"\n📌 {memo}" if memo else ""
    return (
        f"💵 บันทึกรายรับแล้ว!\n"
        f"✅ ยอด: {amount:,.2f} บาท\n"
        f"📂 หมวด: {category}{memo_line}"
    )


def build_budget_reply(amount: float, category: str, period: str) -> str:
    emoji = CATEGORY_EMOJI.get(category, "📝")
    period_th = PERIOD_TH.get(period, period)
    return (
        f"💰 ตั้งงบประมาณแล้ว!\n"
        f"{emoji} หมวด: {category}\n"
        f"💵 งบ: {amount:,.2f} บาท/{period_th}"
    )


def build_summary_reply(summary: dict[str, Any]) -> str:
    month = summary["month"]
    total_income = summary["total_income"]
    total_expense = summary["total_expense"]
    expense_by_cat = summary["expense_by_category"]
    budgets = summary["budgets"]
    balance = total_income - total_expense
    balance_emoji = "✅" if balance >= 0 else "⚠️"

    lines = [
        f"📊 สรุปการเงิน {month}",
        "━━━━━━━━━━━━━━",
        f"💵 รายรับ:  +{total_income:,.0f} บาท",
        f"💸 รายจ่าย: -{total_expense:,.0f} บาท",
        f"{balance_emoji} คงเหลือ:   {balance:,.0f} บาท",
        "━━━━━━━━━━━━━━",
        "📂 รายจ่ายแต่ละหมวด:",
    ]

    if not expense_by_cat:
        lines.append("  ยังไม่มีรายจ่ายเดือนนี้ 🎉")
    else:
        for cat, spent in sorted(expense_by_cat.items(), key=lambda x: -x[1]):
            emoji = CATEGORY_EMOJI.get(cat, "📝")
            budget = budgets.get(cat)
            if budget:
                pct = (spent / budget) * 100
                bar = "🔴" if pct > 100 else "🟡" if pct > 80 else "🟢"
                lines.append(f"  {bar}{emoji} {cat}: {spent:,.0f}/{budget:,.0f} บ ({pct:.0f}%)")
            else:
                lines.append(f"  {emoji} {cat}: {spent:,.0f} บาท")

    if budgets:
        # Show categories with budget but no spending
        unspent = {k: v for k, v in budgets.items() if k not in expense_by_cat}
        for cat, budget in unspent.items():
            emoji = CATEGORY_EMOJI.get(cat, "📝")
            lines.append(f"  🟢{emoji} {cat}: 0/{budget:,.0f} บ (0%)")

    return "\n".join(lines)


# ===========================================================================
# CORE EVENT HANDLER
# ===========================================================================

async def handle_text_event(event: MessageEvent) -> str:
    line_user_id: str = event.source.user_id
    user_text: str = event.message.text.strip()
    lower_text = user_text.lower()

    # Ensure user exists in DB
    await upsert_user(line_user_id, display_name=line_user_id)
    user = await get_user(line_user_id)
    group_id: str | None = user.get("group_id") if user else None

    # -----------------------------------------------------------------------
    # COMMAND: /create — สร้างกลุ่มใหม่
    # -----------------------------------------------------------------------
    if lower_text == "/create":
        if group_id:
            return (
                "⚠️ คุณอยู่ในกลุ่มแล้วนะครับ\n"
                "ไม่สามารถสร้างกลุ่มใหม่ได้ขณะอยู่ในกลุ่ม"
            )
        try:
            group = await create_group(line_user_id)
        except Exception as exc:
            logger.error("Create group error: %s", exc)
            return "⚠️ สร้างกลุ่มไม่สำเร็จ กรุณาลองใหม่นะครับ"

        return (
            f"🎉 สร้างกลุ่มสำเร็จแล้ว!\n"
            f"━━━━━━━━━━━━━━\n"
            f"🔑 รหัสเชิญ: {group['invite_code']}\n"
            f"━━━━━━━━━━━━━━\n"
            f"📤 ส่งรหัสนี้ให้แฟน แล้วให้พิมพ์:\n"
            f"/join {group['invite_code']}"
        )

    # -----------------------------------------------------------------------
    # COMMAND: /join <code> — เข้าร่วมกลุ่ม
    # -----------------------------------------------------------------------
    if lower_text.startswith("/join"):
        parts = user_text.split()
        if len(parts) < 2:
            return "❓ กรุณาระบุรหัสกลุ่มด้วยครับ\nตัวอย่าง: /join ABC123"
        if group_id:
            return "⚠️ คุณอยู่ในกลุ่มแล้วนะครับ"

        invite_code = parts[1].strip().upper()
        try:
            group = await join_group(line_user_id, invite_code)
        except Exception as exc:
            logger.error("Join group error: %s", exc)
            return "⚠️ เข้าร่วมกลุ่มไม่สำเร็จ กรุณาลองใหม่นะครับ"

        if not group:
            return (
                f"❌ ไม่พบรหัสกลุ่ม '{invite_code}'\n"
                "กรุณาตรวจสอบรหัสอีกครั้งนะครับ"
            )

        return (
            f"🎊 เข้าร่วมกลุ่มสำเร็จแล้ว!\n"
            f"👫 ตอนนี้คุณและแฟนอยู่ในกลุ่มเดียวกันแล้ว\n"
            f"💬 เริ่มบันทึกรายรับ-รายจ่ายได้เลยครับ!"
        )

    # -----------------------------------------------------------------------
    # COMMAND: /help — คู่มือการใช้งาน
    # -----------------------------------------------------------------------
    if lower_text in ("/help", "/start"):
        return (
            "🤖 FinCouple AI — คู่มือการใช้งาน\n"
            "━━━━━━━━━━━━━━\n"
            "📌 คำสั่ง:\n"
            "  /create — สร้างกลุ่มใหม่\n"
            "  /join <รหัส> — เข้าร่วมกลุ่ม\n"
            "  /help — ดูคำสั่งทั้งหมด\n"
            "━━━━━━━━━━━━━━\n"
            "💬 พิมพ์ตามธรรมชาติ เช่น:\n"
            "  • 'กินข้าวไป 150 บาท'\n"
            "  • 'เงินเดือนออก 30000'\n"
            "  • 'ตั้งงบค่าอาหาร 5000 บาท'\n"
            "  • 'ขอดูสรุปยอดเดือนนี้'"
        )

    # -----------------------------------------------------------------------
    # ตรวจสอบ group membership ก่อนทำรายการ
    # -----------------------------------------------------------------------
    if not group_id:
        return (
            "👋 สวัสดี! ยังไม่ได้อยู่ในกลุ่มนะครับ\n"
            "━━━━━━━━━━━━━━\n"
            "📌 พิมพ์ /create เพื่อสร้างกลุ่มใหม่\n"
            "📌 พิมพ์ /join <รหัส> เพื่อเข้าร่วมกลุ่ม"
        )

    # -----------------------------------------------------------------------
    # Parse ด้วย Claude AI
    # -----------------------------------------------------------------------
    try:
        parsed = await parse_with_claude(user_text)
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error("Claude parse error: %s", exc)
        return "⚠️ AI ประมวลผลไม่ได้ กรุณาลองใหม่อีกครั้งนะครับ"

    intent: str = parsed.get("intent", "unknown")
    data: dict | None = parsed.get("data")
    error_msg: str | None = parsed.get("error_message")

    # -----------------------------------------------------------------------
    # INTENT: record_expense / record_income
    # -----------------------------------------------------------------------
    if intent in ("record_expense", "record_income"):
        if not data or data.get("amount") is None:
            return f"❓ {error_msg or 'ไม่พบยอดเงินในข้อความ กรุณาระบุจำนวนเงินด้วยครับ'}"

        amount: float = float(data["amount"])
        category: str = data.get("category") or "other"
        memo: str | None = data.get("memo")
        tx_type = "expense" if intent == "record_expense" else "income"

        try:
            await insert_transaction(
                group_id=group_id,
                created_by=line_user_id,
                tx_type=tx_type,
                amount=amount,
                category=category,
                memo=memo,
            )
        except Exception as exc:
            logger.error("Supabase insert error: %s", exc)
            return "⚠️ บันทึกข้อมูลไม่สำเร็จ กรุณาลองใหม่อีกครั้งนะครับ"

        return (
            build_expense_reply(amount, category, memo)
            if tx_type == "expense"
            else build_income_reply(amount, category, memo)
        )

    # -----------------------------------------------------------------------
    # INTENT: set_budget — ตั้งงบประมาณ
    # -----------------------------------------------------------------------
    elif intent == "set_budget":
        if not data or data.get("amount") is None:
            return f"❓ {error_msg or 'ไม่พบยอดงบประมาณ กรุณาระบุจำนวนเงินด้วยครับ'}"

        amount = float(data["amount"])
        category = data.get("category") or "other"
        period = data.get("target_period") or "monthly"
        if period not in ("monthly", "weekly"):
            period = "monthly"

        try:
            await upsert_budget(group_id, category, amount, period)
        except Exception as exc:
            logger.error("Budget upsert error: %s", exc)
            return "⚠️ บันทึกงบประมาณไม่สำเร็จ กรุณาลองใหม่อีกครั้งนะครับ"

        return build_budget_reply(amount, category, period)

    # -----------------------------------------------------------------------
    # INTENT: ask_summary — ดูสรุปยอด
    # -----------------------------------------------------------------------
    elif intent == "ask_summary":
        try:
            summary = await get_monthly_summary(group_id)
        except Exception as exc:
            logger.error("Summary error: %s", exc)
            return "⚠️ ดึงข้อมูลสรุปไม่ได้ กรุณาลองใหม่อีกครั้งนะครับ"

        return build_summary_reply(summary)

    # -----------------------------------------------------------------------
    # INTENT: unknown
    # -----------------------------------------------------------------------
    else:
        return f"❓ {error_msg or 'ไม่เข้าใจข้อความ กรุณาลองพิมพ์ใหม่นะครับ'}"


# ===========================================================================
# FASTAPI ENDPOINTS
# ===========================================================================

@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "FinCouple AI", "version": "4.0.0"})


@app.get("/api/summary")
async def api_summary(line_user_id: str = Query(...)) -> JSONResponse:
    """LIFF Dashboard endpoint — returns monthly summary for the user's group."""
    user = await get_user(line_user_id)
    if not user or not user.get("group_id"):
        raise HTTPException(status_code=404, detail="User not found or not in a group")

    try:
        summary = await get_monthly_summary(user["group_id"])
    except Exception as exc:
        logger.error("API summary error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch summary")

    return JSONResponse(summary)


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
