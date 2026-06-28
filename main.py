"""
FinCouple AI — LINE Bot Webhook Backend
Phase 3: /create, /join, set_budget, ask_summary

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
from zoneinfo import ZoneInfo
from typing import Any

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query, Request

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    FlexMessage,
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
LIFF_URL: str = os.getenv("LIFF_URL", "https://liff.line.me/2010520479-6TrRjatU")

# Timezone
_BKK = ZoneInfo("Asia/Bangkok")

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


app = FastAPI(title="FinCouple AI Webhook", version="3.0.0", lifespan=lifespan)

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
    try:
        result = (
            supabase.table("users")
            .select("line_user_id, display_name, group_id")
            .eq("line_user_id", line_user_id)
            .maybe_single()
            .execute()
        )
        return result.data
    except Exception as exc:
        logger.warning("get_user error (returning None): %s", exc)
        return None


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




async def delete_budget(group_id: str, category: str) -> bool:
    """ลบงบประมาณของหมวดหมู่นั้น — คืน True ถ้าลบได้, False ถ้าไม่มีงบ"""
    result = (
        supabase.table("budgets")
        .select("id")
        .eq("group_id", group_id)
        .eq("category", category)
        .execute()
    )
    if not result.data:
        return False
    supabase.table("budgets").delete().eq("group_id", group_id).eq("category", category).execute()
    return True


async def delete_last_transaction(group_id: str) -> dict[str, Any] | None:
    """ลบรายการล่าสุดของกลุ่ม — คืน row ที่ถูกลบ หรือ None ถ้าไม่มีรายการ"""
    result = (
        supabase.table("transactions")
        .select("id, type, amount, category, memo, created_at")
        .eq("group_id", group_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    last = result.data[0]
    supabase.table("transactions").delete().eq("id", last["id"]).execute()
    return last


async def delete_current_month_transactions(group_id: str) -> int:
    """ลบธุรกรรมทั้งหมดในเดือนปัจจุบัน — คืนจำนวนรายการที่ลบ"""
    now_bkk = datetime.now(_BKK)
    month_start = now_bkk.replace(day=1, hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).isoformat()
    result = (
        supabase.table("transactions")
        .select("id")
        .eq("group_id", group_id)
        .gte("created_at", month_start)
        .execute()
    )
    count = len(result.data) if result.data else 0
    if count > 0:
        supabase.table("transactions").delete().eq("group_id", group_id).gte("created_at", month_start).execute()
    return count

async def get_monthly_summary(group_id: str, month_offset: int = 0) -> dict[str, Any]:
    """Query transactions + budgets for a given month (month_offset=0 current, -1 last month)."""
    _THAI_MONTHS = ["", "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
                    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
    now_bkk = datetime.now(_BKK)

    # Calculate target month in BKK time
    target_month = now_bkk.month + month_offset
    target_year = now_bkk.year
    while target_month <= 0:
        target_month += 12
        target_year -= 1

    month_start_bkk = now_bkk.replace(year=target_year, month=target_month, day=1,
                                       hour=0, minute=0, second=0, microsecond=0)
    next_m = target_month + 1
    next_y = target_year
    if next_m > 12:
        next_m = 1
        next_y += 1
    month_end_bkk = month_start_bkk.replace(year=next_y, month=next_m, day=1)

    month_start = month_start_bkk.astimezone(timezone.utc).isoformat()
    month_end = month_end_bkk.astimezone(timezone.utc).isoformat()

    # Transactions in range
    tx_result = (
        supabase.table("transactions")
        .select("type, amount, category, memo, created_at")
        .eq("group_id", group_id)
        .gte("created_at", month_start)
        .lt("created_at", month_end)
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

    thai_month = _THAI_MONTHS[target_month]
    buddhist_year = target_year + 543
    month_label = f"{thai_month} {buddhist_year}"

    return {
        "month": month_label,
        "total_income": total_income,
        "total_expense": total_expense,
        "balance": total_income - total_expense,
        "expense_by_category": expense_by_cat,
        "budgets": budgets,
        "recent_transactions": transactions[:20],
    }



async def check_budget_alert(group_id: str, category: str) -> dict | None:
    """คืน alert dict ถ้าใช้งบ >= 80% ของเดือนนี้, None ถ้าไม่มีงบหรือยังไม่ถึง threshold"""
    now_bkk = datetime.now(_BKK)
    month_start = now_bkk.replace(day=1, hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).isoformat()

    budget_row = (
        supabase.table("budgets")
        .select("amount")
        .eq("group_id", group_id)
        .eq("category", category)
        .eq("period", "monthly")
        .maybe_single()
        .execute()
    )
    if not budget_row.data:
        return None

    budget_amount = float(budget_row.data["amount"])
    if budget_amount <= 0:
        return None

    spent_rows = (
        supabase.table("transactions")
        .select("amount")
        .eq("group_id", group_id)
        .eq("type", "expense")
        .eq("category", category)
        .gte("created_at", month_start)
        .execute()
    )
    spent = sum(float(r["amount"]) for r in (spent_rows.data or []))
    pct = spent / budget_amount

    if pct < 0.8:
        return None

    return {"spent": spent, "budget": budget_amount, "pct": pct}


async def update_last_transaction_amount(group_id: str, new_amount: float) -> dict | None:
    """แก้ไขจำนวนเงินรายการล่าสุด — คืน row ที่อัปเดต หรือ None ถ้าไม่มีรายการ"""
    result = (
        supabase.table("transactions")
        .select("id, type, amount, category, memo, created_at")
        .eq("group_id", group_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None

    tx = result.data[0]
    old_amount = float(tx["amount"])
    supabase.table("transactions").update({"amount": new_amount}).eq("id", tx["id"]).execute()
    tx["old_amount"] = old_amount
    tx["amount"] = new_amount
    return tx


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
# REPLY BUILDERS — Wealth Emerald Theme (วินัย Bot)
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

_CAT_TH: dict[str, str] = {
    "food": "อาหารและเครื่องดื่ม",
    "travel": "การเดินทาง",
    "home": "ที่อยู่อาศัย",
    "shopping": "ช้อปปิ้ง",
    "entertainment": "บันเทิง",
    "savings": "เงินออม",
    "income": "รายรับ",
    "other": "อื่นๆ",
}

# ── Wealth Emerald Color Palette ────────────────────────────────────────────
_C_BG_PRIMARY   = "#052E16"   # Deep Forest (พื้นหลังหลัก)
_C_BG_SECONDARY = "#064E3B"   # Forest Dark (หัว card)
_C_BG_CARD      = "#0A3D2B"   # Card surface
_C_EMERALD      = "#10B981"   # Primary accent
_C_MINT         = "#34D399"   # Light accent
_C_MINT_DIM     = "#6EE7B7"   # Dimmed mint text
_C_MINT_WHITE   = "#D1FAE5"   # Near-white
_C_AMBER        = "#F59E0B"   # Warm gold accent
_C_INCOME       = "#22C55E"   # รายรับ green
_C_EXPENSE      = "#F87171"   # รายจ่าย coral
_C_DIVIDER      = "#065F46"   # Divider / progress bar bg
_C_WHITE        = "#FFFFFF"


_THAI_MONTHS: dict[str, str] = {
    "January": "มกราคม", "February": "กุมภาพันธ์", "March": "มีนาคม",
    "April": "เมษายน", "May": "พฤษภาคม", "June": "มิถุนายน",
    "July": "กรกฎาคม", "August": "สิงหาคม", "September": "กันยายน",
    "October": "ตุลาคม", "November": "พฤศจิกายน", "December": "ธันวาคม",
}


def _thai_month(month_str: str) -> str:
    parts = month_str.split(" ", 1)
    return f"{_THAI_MONTHS.get(parts[0], parts[0])} {parts[1]}" if len(parts) == 2 else month_str


def _build_category_row(cat: str, spent: float, budget: float | None) -> dict:
    emoji = CATEGORY_EMOJI.get(cat, "📝")
    cat_th = _CAT_TH.get(cat, cat)

    if budget and budget > 0:
        pct = min(int((spent / budget) * 100), 100)
        overspent = spent > budget
        bar_color = _C_EXPENSE if overspent else (_C_AMBER if pct > 80 else _C_EMERALD)
        amount_color = _C_EXPENSE if overspent else (_C_AMBER if pct > 80 else _C_MINT)
        filled_flex = max(pct, 1)
        empty_flex = 100 - filled_flex
        amount_label = f"฿{spent:,.0f} / ฿{budget:,.0f}  {pct}%"
    else:
        bar_color = _C_EMERALD
        amount_color = _C_MINT
        filled_flex = 0
        empty_flex = 100
        amount_label = f"฿{spent:,.0f}"

    row: dict = {
        "type": "box",
        "layout": "vertical",
        "spacing": "sm",
        "contents": [
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": f"{emoji}  {cat_th}", "size": "sm", "color": _C_MINT_WHITE, "flex": 1},
                    {"type": "text", "text": amount_label, "size": "xs", "color": amount_color, "align": "end"},
                ],
            }
        ],
    }

    if budget:
        bar_contents: list = []
        if filled_flex > 0:
            bar_contents.append({
                "type": "box", "layout": "horizontal", "flex": filled_flex,
                "backgroundColor": bar_color, "cornerRadius": "3px", "contents": [],
            })
        if empty_flex > 0:
            bar_contents.append({
                "type": "box", "layout": "horizontal", "flex": empty_flex, "contents": [],
            })
        row["contents"].append({
            "type": "box", "layout": "horizontal", "height": "6px",
            "backgroundColor": _C_DIVIDER, "cornerRadius": "3px", "contents": bar_contents,
        })

    return row


def build_edit_flex(tx: dict) -> dict:
    """Flex Message สำหรับ /แก้ไข — แสดงยอดเดิมและยอดใหม่"""
    old_amount = float(tx.get("old_amount", 0))
    new_amount = float(tx["amount"])
    tx_type = tx.get("type", "expense")
    type_color = _C_EXPENSE if tx_type == "expense" else _C_INCOME
    cat = tx.get("category", "other")
    emoji = CATEGORY_EMOJI.get(cat, "📝")
    cat_th = _CAT_TH.get(cat, cat)
    type_label = "รายจ่าย" if tx_type == "expense" else "รายรับ"

    contents = {
        "type": "bubble", "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "vertical",
                    "backgroundColor": "#1C1917",
                    "paddingTop": "14px", "paddingBottom": "10px",
                    "paddingStart": "16px", "paddingEnd": "16px",
                    "contents": [
                        {"type": "text", "text": "✏️ แก้ไขรายการแล้วครับ",
                         "size": "sm", "color": "#FFFFFF", "weight": "bold"},
                    ],
                },
                {"type": "box", "layout": "horizontal", "height": "2px",
                 "backgroundColor": _C_MINT, "contents": []},
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY,
                    "paddingAll": "16px", "spacing": "sm",
                    "contents": [
                        {"type": "box", "layout": "horizontal", "contents": [
                            {"type": "text", "text": "ยอดเดิม", "size": "sm",
                             "color": _C_MINT_DIM, "flex": 3},
                            {"type": "text", "text": f"฿{old_amount:,.2f}", "size": "sm",
                             "color": _C_MINT_DIM, "flex": 4, "decoration": "line-through", "align": "end"},
                        ]},
                        {"type": "box", "layout": "horizontal", "contents": [
                            {"type": "text", "text": "ยอดใหม่", "size": "md",
                             "color": _C_MINT_WHITE, "weight": "bold", "flex": 3},
                            {"type": "text", "text": f"฿{new_amount:,.2f}", "size": "md",
                             "color": type_color, "weight": "bold", "flex": 4, "align": "end"},
                        ]},
                        {"type": "separator", "margin": "sm", "color": _C_BG_SECONDARY},
                        {"type": "text", "text": f"{emoji}  {cat_th}  ·  {type_label}",
                         "size": "xs", "color": _C_MINT_DIM, "margin": "sm"},
                    ],
                },
            ],
        },
    }
    return {"alt_text": f"แก้ไขรายการเป็น ฿{new_amount:,.0f}", "contents": contents}


def build_expense_flex(amount: float, category: str, memo: str | None, alert_data: dict | None = None) -> dict:
    emoji = CATEGORY_EMOJI.get(category, "📝")
    cat_th = _CAT_TH.get(category, category)
    body_contents: list = [
        {"type": "text", "text": f"฿{amount:,.2f}", "size": "3xl", "color": _C_EXPENSE, "weight": "bold"},
    ]
    if memo:
        body_contents.append({"type": "text", "text": f"📌 {memo}", "size": "sm", "color": _C_MINT_DIM, "margin": "sm"})
    body_contents.append({
        "type": "box", "layout": "horizontal", "margin": "md",
        "contents": [{
            "type": "box", "layout": "horizontal",
            "backgroundColor": _C_DIVIDER, "cornerRadius": "20px",
            "paddingTop": "4px", "paddingBottom": "4px",
            "paddingStart": "10px", "paddingEnd": "10px",
            "contents": [{"type": "text", "text": f"{emoji}  {cat_th}", "size": "xs", "color": _C_MINT}],
        }],
    })
    # Budget alert box (append to body_contents if needed)
    if alert_data:
        pct = alert_data["pct"]
        spent = alert_data["spent"]
        budget = alert_data["budget"]
        remaining = budget - spent
        if pct >= 1.0:
            alert_icon = "🚨"
            alert_text = f"ใช้งบเกินแล้ว! ใช้ไป ฿{spent:,.0f} / ฿{budget:,.0f}"
            alert_color = _C_EXPENSE
        else:
            alert_icon = "⚠️"
            alert_text = f"ใช้งบไปแล้ว {pct*100:.0f}% — เหลือ ฿{remaining:,.0f}"
            alert_color = _C_AMBER
        body_contents.append({
            "type": "box", "layout": "horizontal",
            "backgroundColor": "#1C0A00", "cornerRadius": "6px",
            "paddingAll": "8px", "margin": "md",
            "contents": [
                {"type": "text", "text": alert_icon, "size": "sm", "flex": 0},
                {"type": "text", "text": alert_text, "size": "xxs",
                 "color": alert_color, "flex": 1, "margin": "sm", "wrap": True},
            ],
        })

    contents = {
        "type": "bubble", "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "horizontal",
                    "backgroundColor": _C_BG_SECONDARY,
                    "paddingTop": "16px", "paddingBottom": "12px",
                    "paddingStart": "16px", "paddingEnd": "16px", "alignItems": "center",
                    "contents": [
                        {"type": "text", "text": emoji, "size": "xl", "flex": 0},
                        {"type": "box", "layout": "vertical", "flex": 1, "margin": "sm",
                         "contents": [
                             {"type": "text", "text": "บันทึกรายจ่ายแล้วครับ", "size": "sm", "color": _C_WHITE, "weight": "bold"},
                             {"type": "text", "text": cat_th, "size": "xs", "color": _C_MINT, "margin": "xs"},
                         ]},
                    ],
                },
                {"type": "box", "layout": "horizontal", "height": "2px", "backgroundColor": _C_EXPENSE, "contents": []},
                {"type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY, "paddingAll": "16px", "contents": body_contents},
            ],
        },
    }
    return {"alt_text": f"บันทึกรายจ่าย ฿{amount:,.0f} บาท", "contents": contents}


def build_income_flex(amount: float, category: str, memo: str | None) -> dict:
    body_contents: list = [
        {"type": "text", "text": f"+฿{amount:,.2f}", "size": "3xl", "color": _C_INCOME, "weight": "bold"},
    ]
    if memo:
        body_contents.append({"type": "text", "text": f"📌 {memo}", "size": "sm", "color": _C_MINT_DIM, "margin": "sm"})
    body_contents.append({
        "type": "box", "layout": "horizontal", "margin": "md",
        "contents": [{
            "type": "box", "layout": "horizontal",
            "backgroundColor": _C_BG_CARD, "cornerRadius": "20px",
            "paddingTop": "4px", "paddingBottom": "4px",
            "paddingStart": "10px", "paddingEnd": "10px",
            "borderColor": _C_INCOME, "borderWidth": "1px",
            "contents": [{"type": "text", "text": "💵  รายรับ", "size": "xs", "color": _C_INCOME}],
        }],
    })
    contents = {
        "type": "bubble", "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "horizontal",
                    "backgroundColor": _C_BG_CARD,
                    "paddingTop": "16px", "paddingBottom": "12px",
                    "paddingStart": "16px", "paddingEnd": "16px", "alignItems": "center",
                    "contents": [
                        {"type": "text", "text": "💵", "size": "xl", "flex": 0},
                        {"type": "box", "layout": "vertical", "flex": 1, "margin": "sm",
                         "contents": [
                             {"type": "text", "text": "บันทึกรายรับแล้วครับ", "size": "sm", "color": _C_WHITE, "weight": "bold"},
                             {"type": "text", "text": "รายรับ", "size": "xs", "color": _C_INCOME, "margin": "xs"},
                         ]},
                    ],
                },
                {"type": "box", "layout": "horizontal", "height": "2px", "backgroundColor": _C_INCOME, "contents": []},
                {"type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY, "paddingAll": "16px", "contents": body_contents},
            ],
        },
    }
    return {"alt_text": f"บันทึกรายรับ ฿{amount:,.0f} บาท", "contents": contents}


def build_budget_flex(amount: float, category: str, period: str) -> dict:
    emoji = CATEGORY_EMOJI.get(category, "📝")
    cat_th = _CAT_TH.get(category, category)
    period_th = {"monthly": "รายเดือน", "weekly": "รายสัปดาห์", "current_month": "รายเดือน"}.get(period, period)
    contents = {
        "type": "bubble", "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "vertical",
                    "backgroundColor": _C_BG_SECONDARY,
                    "paddingTop": "16px", "paddingBottom": "12px",
                    "paddingStart": "16px", "paddingEnd": "16px",
                    "contents": [{"type": "text", "text": "💰 ตั้งงบประมาณแล้วครับ", "size": "sm", "color": _C_WHITE, "weight": "bold"}],
                },
                {"type": "box", "layout": "horizontal", "height": "2px", "backgroundColor": _C_AMBER, "contents": []},
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY,
                    "paddingAll": "16px", "spacing": "sm",
                    "contents": [
                        {"type": "text", "text": f"{emoji}  {cat_th}", "size": "md", "color": _C_MINT_WHITE, "weight": "bold"},
                        {"type": "text", "text": f"฿{amount:,.2f} / {period_th}", "size": "xxl", "color": _C_AMBER, "weight": "bold", "margin": "sm"},
                    ],
                },
            ],
        },
    }
    return {"alt_text": f"ตั้งงบ {cat_th} ฿{amount:,.0f}", "contents": contents}


def build_delete_budget_flex(category: str, cat_th: str, emoji: str) -> dict:
    """Flex Message สำหรับ /ลบงบ"""
    contents = {
        "type": "bubble", "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "vertical",
                    "backgroundColor": "#78350F",
                    "paddingTop": "14px", "paddingBottom": "10px",
                    "paddingStart": "16px", "paddingEnd": "16px",
                    "contents": [{"type": "text", "text": "🗑️ ลบงบประมาณแล้วครับ", "size": "sm", "color": "#FFFFFF", "weight": "bold"}],
                },
                {"type": "box", "layout": "horizontal", "height": "2px", "backgroundColor": _C_AMBER, "contents": []},
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY,
                    "paddingAll": "16px", "spacing": "sm",
                    "contents": [
                        {"type": "text", "text": f"{emoji}  {cat_th}", "size": "md", "color": _C_MINT_WHITE, "weight": "bold"},
                        {"type": "text", "text": "งบหมวดนี้ถูกลบออกจากระบบแล้ว", "size": "sm", "color": _C_MINT_DIM, "margin": "sm"},
                        {"type": "text", "text": "ตั้งงบใหม่ได้โดยพิมพ์จำนวนเงิน + หมวดหมู่ครับ", "size": "xxs", "color": _C_MINT_DIM, "wrap": True},
                    ],
                },
            ],
        },
    }
    return {"alt_text": f"ลบงบ {cat_th}", "contents": contents}


def build_delete_flex(deleted: dict) -> dict:
    """Flex Message สำหรับ /ลบล่าสุด"""
    cat_emoji = {
        "food": "🍜", "travel": "🚗", "home": "🏠", "shopping": "🛍️",
        "entertainment": "🎬", "savings": "💰", "income": "💵", "other": "📝",
    }
    tx_type_label = "รายจ่าย" if deleted["type"] == "expense" else "รายรับ"
    type_color = _C_EXPENSE if deleted["type"] == "expense" else _C_INCOME
    emoji = cat_emoji.get(deleted.get("category", "other"), "📝")
    amount = deleted["amount"]
    memo = deleted.get("memo") or deleted.get("category") or "-"
    contents = {
        "type": "bubble", "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "vertical",
                    "backgroundColor": "#7F1D1D",
                    "paddingTop": "14px", "paddingBottom": "10px",
                    "paddingStart": "16px", "paddingEnd": "16px",
                    "contents": [{"type": "text", "text": "🗑️ ลบรายการล่าสุดแล้วครับ", "size": "sm", "color": "#FFFFFF", "weight": "bold"}],
                },
                {"type": "box", "layout": "horizontal", "height": "2px", "backgroundColor": _C_EXPENSE, "contents": []},
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY,
                    "paddingAll": "16px", "spacing": "sm",
                    "contents": [
                        {"type": "text", "text": f"{emoji}  {memo}", "size": "md", "color": _C_MINT_WHITE, "weight": "bold"},
                        {
                            "type": "box", "layout": "horizontal", "margin": "sm",
                            "contents": [
                                {"type": "text", "text": tx_type_label, "size": "sm", "color": type_color, "flex": 1},
                                {"type": "text", "text": f"฿{amount:,.0f}", "size": "lg", "color": type_color, "weight": "bold", "align": "end"},
                            ],
                        },
                    ],
                },
            ],
        },
    }
    return {"alt_text": f"ลบรายการ {memo} ฿{amount:,.0f}", "contents": contents}


def build_clearmonth_flex(count: int, month_name: str) -> dict:
    """Flex Message สำหรับ /ล้างเดือน"""
    contents = {
        "type": "bubble", "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "vertical",
                    "backgroundColor": "#1E1B4B",
                    "paddingTop": "14px", "paddingBottom": "10px",
                    "paddingStart": "16px", "paddingEnd": "16px",
                    "contents": [{"type": "text", "text": "✅ ล้างข้อมูลเรียบร้อยแล้วครับ", "size": "sm", "color": "#FFFFFF", "weight": "bold"}],
                },
                {"type": "box", "layout": "horizontal", "height": "2px", "backgroundColor": _C_AMBER, "contents": []},
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY,
                    "paddingAll": "16px", "spacing": "sm",
                    "contents": [
                        {
                            "type": "box", "layout": "horizontal",
                            "contents": [
                                {"type": "text", "text": "🗑️ ลบทั้งหมด", "size": "sm", "color": _C_MINT_DIM, "flex": 1},
                                {"type": "text", "text": f"{count} รายการ", "size": "sm", "color": _C_MINT_WHITE, "weight": "bold", "align": "end"},
                            ],
                        },
                        {
                            "type": "box", "layout": "horizontal", "margin": "sm",
                            "contents": [
                                {"type": "text", "text": "📅 เดือน", "size": "sm", "color": _C_MINT_DIM, "flex": 1},
                                {"type": "text", "text": month_name, "size": "sm", "color": _C_MINT_WHITE, "weight": "bold", "align": "end"},
                            ],
                        },
                    ],
                },
            ],
        },
    }
    return {"alt_text": f"ล้างข้อมูล {count} รายการ เดือน {month_name}", "contents": contents}


def build_help_flex() -> dict:
    def _cmd(name: str, desc: str) -> dict:
        return {
            "type": "box", "layout": "vertical",
            "paddingTop": "10px", "paddingBottom": "10px",
            "paddingStart": "16px", "paddingEnd": "16px",
            "contents": [
                {"type": "text", "text": name, "size": "sm", "color": _C_AMBER, "weight": "bold"},
                {"type": "text", "text": desc, "size": "xs", "color": _C_MINT_DIM, "margin": "xs"},
            ],
        }
    contents = {
        "type": "bubble", "size": "mega",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_SECONDARY,
                    "paddingTop": "18px", "paddingBottom": "14px",
                    "paddingStart": "18px", "paddingEnd": "18px",
                    "contents": [
                        {"type": "text", "text": "💎 วินัย", "size": "xl", "color": _C_AMBER, "weight": "bold"},
                        {"type": "text", "text": "ผู้ช่วยดูแลการเงินคู่รัก", "size": "sm", "color": _C_MINT, "margin": "xs"},
                    ],
                },
                {"type": "box", "layout": "horizontal", "height": "2px", "backgroundColor": _C_EMERALD, "contents": []},
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY,
                    "paddingTop": "4px", "paddingBottom": "0px",
                    "contents": [
                        _cmd("/create", "สร้างกลุ่มใหม่สำหรับคุณและแฟน"),
                        {"type": "box", "height": "1px", "layout": "horizontal", "backgroundColor": _C_DIVIDER, "marginStart": "16px", "contents": []},
                        _cmd("/join <รหัส>", "เข้าร่วมกลุ่มด้วยรหัสเชิญ"),
                        {"type": "box", "height": "1px", "layout": "horizontal", "backgroundColor": _C_DIVIDER, "marginStart": "16px", "contents": []},
                        _cmd("พิมพ์ตามธรรมชาติ", "เช่น 'กินข้าวไป 150 บาท' หรือ 'เงินเดือนออก 30000'"),
                        {"type": "box", "height": "1px", "layout": "horizontal", "backgroundColor": _C_DIVIDER, "marginStart": "16px", "contents": []},
                        _cmd("สรุป / ขอดูสรุป", "ดูสรุปรายรับ-รายจ่ายประจำเดือน"),
                    ],
                },
                {
                    "type": "box", "layout": "horizontal", "backgroundColor": _C_BG_CARD, "paddingAll": "12px",
                    "contents": [{"type": "text",
                                  "text": "💡 กดปุ่ม บันทึก เพื่อเริ่มต้น หรือ Dashboard เพื่อดูภาพรวมครับ",
                                  "size": "xs", "color": _C_MINT_DIM, "wrap": True}],
                },
            ],
        },
    }
    return {"alt_text": "คู่มือการใช้งาน วินัย", "contents": contents}


def build_create_flex(invite_code: str) -> dict:
    contents = {
        "type": "bubble", "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_SECONDARY,
                    "paddingTop": "16px", "paddingBottom": "12px",
                    "paddingStart": "16px", "paddingEnd": "16px",
                    "contents": [{"type": "text", "text": "🎉 สร้างกลุ่มสำเร็จแล้วครับ!", "size": "sm", "color": _C_WHITE, "weight": "bold"}],
                },
                {"type": "box", "layout": "horizontal", "height": "2px", "backgroundColor": _C_EMERALD, "contents": []},
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY,
                    "paddingAll": "18px", "spacing": "sm",
                    "contents": [
                        {"type": "text", "text": "รหัสเชิญสำหรับแฟนครับ", "size": "xs", "color": _C_MINT_DIM},
                        {
                            "type": "box", "layout": "horizontal",
                            "backgroundColor": _C_BG_CARD, "cornerRadius": "12px",
                            "paddingTop": "14px", "paddingBottom": "14px", "margin": "sm",
                            "justifyContent": "center",
                            "contents": [{"type": "text", "text": invite_code, "size": "3xl", "color": _C_AMBER, "weight": "bold", "align": "center"}],
                        },
                        {"type": "text", "text": "📤 ส่งรหัสนี้ให้แฟน แล้วให้พิมพ์:", "size": "xs", "color": _C_MINT_DIM, "margin": "md"},
                        {
                            "type": "box", "layout": "horizontal",
                            "backgroundColor": _C_DIVIDER, "cornerRadius": "8px",
                            "paddingAll": "8px", "margin": "xs",
                            "contents": [{"type": "text", "text": f"/join {invite_code}", "size": "sm", "color": _C_MINT, "weight": "bold", "align": "center"}],
                        },
                    ],
                },
            ],
        },
    }
    return {"alt_text": f"รหัสเชิญกลุ่ม: {invite_code}", "contents": contents}


def build_join_flex() -> dict:
    contents = {
        "type": "bubble", "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_CARD,
                    "paddingTop": "16px", "paddingBottom": "12px",
                    "paddingStart": "16px", "paddingEnd": "16px",
                    "contents": [{"type": "text", "text": "🎊 เข้าร่วมกลุ่มสำเร็จแล้วครับ!", "size": "sm", "color": _C_WHITE, "weight": "bold"}],
                },
                {"type": "box", "layout": "horizontal", "height": "2px", "backgroundColor": _C_EMERALD, "contents": []},
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY,
                    "paddingAll": "18px", "spacing": "sm",
                    "contents": [
                        {"type": "text", "text": "👫 ตอนนี้คุณและแฟนอยู่ในกลุ่มเดียวกันแล้วครับ", "size": "sm", "color": _C_MINT_WHITE, "wrap": True},
                        {"type": "text", "text": "💬 เริ่มบันทึกรายรับ-รายจ่ายได้เลยครับ!", "size": "sm", "color": _C_MINT, "margin": "md"},
                    ],
                },
            ],
        },
    }
    return {"alt_text": "เข้าร่วมกลุ่มสำเร็จแล้วครับ", "contents": contents}


def build_record_prompt_flex() -> dict:
    def _ex(text: str) -> dict:
        return {
            "type": "box", "layout": "horizontal",
            "backgroundColor": _C_BG_SECONDARY, "cornerRadius": "8px", "paddingAll": "10px",
            "contents": [{"type": "text", "text": text, "size": "sm", "color": _C_MINT_WHITE, "wrap": True}],
        }
    contents = {
        "type": "bubble", "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_SECONDARY,
                    "paddingTop": "16px", "paddingBottom": "12px",
                    "paddingStart": "16px", "paddingEnd": "16px",
                    "contents": [{"type": "text", "text": "✏️ พิมพ์รายการที่ต้องการบันทึกครับ", "size": "sm", "color": _C_WHITE, "weight": "bold"}],
                },
                {"type": "box", "layout": "horizontal", "height": "2px", "backgroundColor": _C_EMERALD, "contents": []},
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY,
                    "paddingAll": "16px", "spacing": "sm",
                    "contents": [
                        {"type": "text", "text": "ตัวอย่างข้อความ:", "size": "xs", "color": _C_MINT_DIM},
                        _ex("🍜  กินข้าวไป 150 บาท"),
                        _ex("💵  เงินเดือนออก 30,000"),
                        _ex("💰  ตั้งงบค่าอาหาร 5,000 บาท"),
                    ],
                },
            ],
        },
    }
    return {"alt_text": "บันทึกรายการ", "contents": contents}


def build_no_group_flex() -> dict:
    contents = {
        "type": "bubble", "size": "kilo",
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "0px",
            "contents": [
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_SECONDARY,
                    "paddingTop": "16px", "paddingBottom": "12px",
                    "paddingStart": "16px", "paddingEnd": "16px",
                    "contents": [
                        {"type": "text", "text": "👋 สวัสดีครับ! วินัยพร้อมช่วยแล้ว", "size": "sm", "color": _C_WHITE, "weight": "bold"},
                        {"type": "text", "text": "สร้างหรือเข้าร่วมกลุ่มก่อนนะครับ", "size": "xs", "color": _C_MINT, "margin": "xs"},
                    ],
                },
                {"type": "box", "layout": "horizontal", "height": "2px", "backgroundColor": _C_AMBER, "contents": []},
                {
                    "type": "box", "layout": "vertical", "backgroundColor": _C_BG_PRIMARY,
                    "paddingAll": "16px", "spacing": "sm",
                    "contents": [
                        {
                            "type": "box", "layout": "horizontal",
                            "backgroundColor": _C_BG_CARD, "cornerRadius": "8px", "paddingAll": "10px",
                            "contents": [
                                {"type": "text", "text": "📌  /create", "size": "sm", "color": _C_EMERALD, "weight": "bold", "flex": 0},
                                {"type": "text", "text": "สร้างกลุ่มใหม่", "size": "sm", "color": _C_MINT_DIM, "margin": "md"},
                            ],
                        },
                        {
                            "type": "box", "layout": "horizontal",
                            "backgroundColor": _C_BG_CARD, "cornerRadius": "8px", "paddingAll": "10px",
                            "contents": [
                                {"type": "text", "text": "📌  /join <รหัส>", "size": "sm", "color": _C_EMERALD, "weight": "bold", "flex": 0},
                                {"type": "text", "text": "เข้าร่วมกลุ่ม", "size": "sm", "color": _C_MINT_DIM, "margin": "md"},
                            ],
                        },
                    ],
                },
            ],
        },
    }
    return {"alt_text": "เริ่มต้นใช้งาน วินัย", "contents": contents}


def build_summary_flex(summary: dict[str, Any]) -> dict:
    """Build a Wealth Emerald Flex Message for the monthly summary."""
    month_th = _thai_month(summary["month"])
    total_income: float = summary["total_income"]
    total_expense: float = summary["total_expense"]
    balance: float = summary["balance"]
    expense_by_cat: dict[str, float] = summary["expense_by_category"]
    budgets: dict[str, float] = summary["budgets"]

    # Merge categories: expenses + budgets with zero spend
    all_cats: dict[str, dict] = {}
    for cat, spent in expense_by_cat.items():
        all_cats[cat] = {"spent": spent, "budget": budgets.get(cat)}
    for cat, bud in budgets.items():
        if cat not in all_cats:
            all_cats[cat] = {"spent": 0.0, "budget": bud}

    # Sort by spent descending, max 6 rows
    sorted_cats = sorted(all_cats.items(), key=lambda x: -x[1]["spent"])
    category_rows: list = [_build_category_row(c, d["spent"], d["budget"]) for c, d in sorted_cats[:6]]

    if not category_rows:
        category_rows = [{
            "type": "text", "text": "ยังไม่มีรายจ่ายเดือนนี้ 🎉",
            "size": "sm", "color": _C_INCOME, "align": "center",
        }]

    balance_color = _C_INCOME if balance >= 0 else _C_EXPENSE

    contents = {
        "type": "bubble",
        "size": "giga",

        "styles": {
            "body": {"backgroundColor": _C_BG_PRIMARY},
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "0px",
            "backgroundColor": _C_BG_PRIMARY,
            "contents": [
                # ── Header ──────────────────────────────────────────────────
                {
                    "type": "box",
                    "layout": "horizontal",
                    "paddingTop": "20px",
                    "paddingBottom": "16px",
                    "paddingStart": "20px",
                    "paddingEnd": "20px",
                    "alignItems": "center",
                    "backgroundColor": _C_BG_SECONDARY,
                    "contents": [
                        {
                            "type": "box",
                            "layout": "vertical",
                            "flex": 1,
                            "contents": [
                                {"type": "text", "text": "✦  วินัย · ผู้ช่วยการเงิน",
                                 "size": "xxs", "color": _C_AMBER, "weight": "bold"},
                                {"type": "text", "text": "สรุปการเงิน",
                                 "size": "xxl", "color": _C_WHITE, "weight": "bold", "margin": "xs"},
                                {"type": "text", "text": month_th,
                                 "size": "sm", "color": _C_MINT, "margin": "xs"},
                            ],
                        },
                        {"type": "text", "text": "💎", "size": "3xl", "align": "end", "flex": 0},
                    ],
                },
                # ── Emerald separator ───────────────────────────────────────
                {"type": "box", "layout": "horizontal", "height": "2px",
                 "backgroundColor": _C_EMERALD, "contents": []},
                # ── Stat cards (Bento) ──────────────────────────────────────
                {
                    "type": "box",
                    "layout": "horizontal",
                    "paddingTop": "16px",
                    "paddingBottom": "8px",
                    "paddingStart": "16px",
                    "paddingEnd": "16px",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "box", "layout": "vertical", "flex": 1,
                            "backgroundColor": _C_BG_CARD, "cornerRadius": "12px",
                            "paddingAll": "12px", "borderColor": _C_INCOME, "borderWidth": "1px",
                            "contents": [
                                {"type": "text", "text": "💚 รายรับ", "size": "xs",
                                 "color": _C_INCOME, "weight": "bold"},
                                {"type": "text", "text": f"฿{total_income:,.0f}",
                                 "size": "lg", "color": _C_WHITE, "weight": "bold", "margin": "sm"},
                            ],
                        },
                        {
                            "type": "box", "layout": "vertical", "flex": 1,
                            "backgroundColor": "#1A0D0D", "cornerRadius": "12px",
                            "paddingAll": "12px", "borderColor": _C_EXPENSE, "borderWidth": "1px",
                            "contents": [
                                {"type": "text", "text": "❤️ รายจ่าย", "size": "xs",
                                 "color": _C_EXPENSE, "weight": "bold"},
                                {"type": "text", "text": f"฿{total_expense:,.0f}",
                                 "size": "lg", "color": _C_WHITE, "weight": "bold", "margin": "sm"},
                            ],
                        },
                        {
                            "type": "box", "layout": "vertical", "flex": 1,
                            "backgroundColor": "#1C1500", "cornerRadius": "12px",
                            "paddingAll": "12px", "borderColor": _C_AMBER, "borderWidth": "1px",
                            "contents": [
                                {"type": "text", "text": "✨ คงเหลือ", "size": "xs",
                                 "color": _C_AMBER, "weight": "bold"},
                                {"type": "text", "text": f"฿{balance:,.0f}",
                                 "size": "lg", "color": balance_color, "weight": "bold", "margin": "sm"},
                            ],
                        },
                    ],
                },
                # ── Budget section label ────────────────────────────────────
                {
                    "type": "box", "layout": "vertical",
                    "paddingTop": "12px", "paddingBottom": "6px",
                    "paddingStart": "16px", "paddingEnd": "16px",
                    "contents": [
                        {"type": "text", "text": "◆  งบประมาณหมวดหมู่",
                         "size": "xs", "color": _C_MINT, "weight": "bold"}
                    ],
                },
                # ── Category rows ───────────────────────────────────────────
                {
                    "type": "box", "layout": "vertical",
                    "paddingStart": "16px", "paddingEnd": "16px", "paddingBottom": "16px",
                    "spacing": "md",
                    "contents": category_rows,
                },
                # ── Bottom divider ──────────────────────────────────────────
                {"type": "box", "layout": "horizontal", "height": "1px",
                 "backgroundColor": _C_DIVIDER, "contents": []},
            ],
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "paddingAll": "16px",
            "backgroundColor": _C_BG_PRIMARY,
            "contents": [
                {
                    "type": "button",
                    "action": {"type": "uri", "label": "📊  เปิด Dashboard แบบเต็ม", "uri": LIFF_URL},
                    "style": "primary",
                    "color": _C_EMERALD,
                    "height": "sm",
                }
            ],
        },
    }

    return {"alt_text": f"📊 สรุปการเงิน วินัย — {summary['month']}", "contents": contents}


# ===========================================================================
# CORE EVENT HANDLER
# ===========================================================================

async def handle_text_event(event: MessageEvent) -> str | dict:
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

        return build_create_flex(group["invite_code"])

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

        return build_join_flex()

    # -----------------------------------------------------------------------
    # COMMAND: บันทึก — prompt ให้พิมพ์รายการ
    # -----------------------------------------------------------------------
    if lower_text == "บันทึก":
        return build_record_prompt_flex()

    # -----------------------------------------------------------------------
    # COMMAND: /help — คู่มือการใช้งาน
    # -----------------------------------------------------------------------
    if lower_text in ("/help", "/start"):
        return build_help_flex()

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # COMMAND: /ลบงบ[หมวด] — ลบงบประมาณของหมวดหมู่นั้น
    # -----------------------------------------------------------------------
    _CAT_KEYWORDS: dict[str, list[str]] = {
        "food":          ["อาหาร", "food", "กิน", "เครื่องดื่ม"],
        "travel":        ["เดินทาง", "travel", "รถ", "น้ำมัน", "ค่ารถ"],
        "home":          ["บ้าน", "home", "ที่อยู่", "ค่าน้ำ", "ค่าไฟ", "เน็ต"],
        "shopping":      ["ช้อปปิ้ง", "shopping", "เสื้อผ้า", "ช้อป"],
        "entertainment": ["บันเทิง", "entertainment", "หนัง", "ท่องเที่ยว"],
        "savings":       ["ออม", "savings", "ลงทุน", "เก็บ"],
        "income":        ["รายรับ", "income", "เงินเดือน"],
        "other":         ["อื่นๆ", "other", "อื่น"],
    }
    if lower_text.startswith("/ลบงบ") or lower_text.startswith("ลบงบ"):
        if not group_id:
            return build_no_group_flex()
        raw_keyword = lower_text.replace("/ลบงบ", "").replace("ลบงบ", "").strip()
        matched_cat: str | None = None
        for cat, keywords in _CAT_KEYWORDS.items():
            if any(kw in raw_keyword for kw in keywords):
                matched_cat = cat
                break
        if not matched_cat:
            lines = ["🗂️ ระบุหมวดหมู่ที่ต้องการลบงบครับ เช่น:"]
            for cat, kws in _CAT_KEYWORDS.items():
                emoji = CATEGORY_EMOJI.get(cat, "📝")
                lines.append(f"  {emoji} /ลบงบ{kws[0]}")
            return "\n".join(lines)
        try:
            deleted_ok = await delete_budget(group_id, matched_cat)
        except Exception as exc:
            logger.error("Delete budget error: %s", exc)
            return "⚠️ ลบงบไม่สำเร็จ กรุณาลองใหม่ครับ"
        if not deleted_ok:
            cat_th = _CAT_TH.get(matched_cat, matched_cat)
            return f"📭 ยังไม่มีงบหมวด {cat_th} ในระบบครับ"
        emoji = CATEGORY_EMOJI.get(matched_cat, "📝")
        cat_th = _CAT_TH.get(matched_cat, matched_cat)
        return build_delete_budget_flex(matched_cat, cat_th, emoji)

    # -----------------------------------------------------------------------
    # COMMAND: /แก้ไข <จำนวน> — แก้ไขยอดรายการล่าสุด
    # -----------------------------------------------------------------------
    if lower_text.startswith("/แก้ไข") or lower_text.startswith("แก้ไข"):
        if not group_id:
            return build_no_group_flex()
        raw_num = lower_text.replace("/แก้ไข", "").replace("แก้ไข", "").strip().replace(",", "")
        try:
            new_amount = float(raw_num)
            if new_amount <= 0:
                raise ValueError
        except ValueError:
            return "❓ ระบุจำนวนเงินที่ต้องการแก้ไขครับ\nตัวอย่าง: /แก้ไข 500"
        try:
            updated = await update_last_transaction_amount(group_id, new_amount)
        except Exception as exc:
            logger.error("Update transaction error: %s", exc)
            return "⚠️ แก้ไขรายการไม่สำเร็จ กรุณาลองใหม่ครับ"
        if not updated:
            return "📭 ยังไม่มีรายการในระบบครับ"
        return build_edit_flex(updated)

    # -----------------------------------------------------------------------
    # COMMAND: /สรุปเดือนที่แล้ว — สรุปยอดเดือนก่อน
    # -----------------------------------------------------------------------
    if lower_text in ("/สรุปเดือนที่แล้ว", "สรุปเดือนที่แล้ว", "/เดือนที่แล้ว", "เดือนที่แล้ว"):
        if not group_id:
            return build_no_group_flex()
        try:
            summary = await get_monthly_summary(group_id, month_offset=-1)
        except Exception as exc:
            logger.error("Last month summary error: %s", exc)
            return "⚠️ ดึงข้อมูลเดือนที่แล้วไม่ได้ กรุณาลองใหม่ครับ"
        return build_summary_flex(summary)

    # COMMAND: /ลบล่าสุด — ลบรายการล่าสุดของกลุ่ม
    # -----------------------------------------------------------------------
    if lower_text in ("/ลบล่าสุด", "/undo", "ลบล่าสุด"):
        if not group_id:
            return build_no_group_flex()
        try:
            deleted = await delete_last_transaction(group_id)
        except Exception as exc:
            logger.error("Delete last tx error: %s", exc)
            return "⚠️ ลบรายการไม่สำเร็จ กรุณาลองใหม่ครับ"

        if not deleted:
            return "📭 ยังไม่มีรายการในระบบครับ"

        return build_delete_flex(deleted)

    # -----------------------------------------------------------------------
    # COMMAND: /ล้างเดือน — ลบธุรกรรมทั้งหมดในเดือนนี้
    # -----------------------------------------------------------------------
    if lower_text in ("/ล้างเดือน", "/clearmonth", "ล้างเดือน"):
        if not group_id:
            return build_no_group_flex()
        _TH_M = ["","มกราคม","กุมภาพันธ์","มีนาคม","เมษายน","พฤษภาคม","มิถุนายน","กรกฎาคม","สิงหาคม","กันยายน","ตุลาคม","พฤศจิกายน","ธันวาคม"]
        now_bkk = datetime.now(_BKK)
        month_name = f"{_TH_M[now_bkk.month]} {now_bkk.year + 543}"
        try:
            count = await delete_current_month_transactions(group_id)
        except Exception as exc:
            logger.error("Clear month error: %s", exc)
            return "⚠️ ล้างข้อมูลไม่สำเร็จ กรุณาลองใหม่ครับ"

        if count == 0:
            return "📭 ไม่มีรายการในเดือนนี้ครับ"
        return build_clearmonth_flex(count, month_name)

    # -----------------------------------------------------------------------
    # ตรวจสอบ group membership ก่อนทำรายการ
    # -----------------------------------------------------------------------
    if not group_id:
        return build_no_group_flex()

    # -----------------------------------------------------------------------
    # Parse ด้วย Claude AI
    # -----------------------------------------------------------------------
    try:
        parsed = await parse_with_claude(user_text)
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error("Claude parse error: %s", exc)
        return "⚠️ วินัยประมวลผลไม่ได้ในขณะนี้ครับ กรุณาลองใหม่อีกครั้งนะครับ"

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

        if tx_type == "expense":
            try:
                alert_data = await check_budget_alert(group_id, category)
            except Exception:
                alert_data = None
            return build_expense_flex(amount, category, memo, alert_data)
        else:
            return build_income_flex(amount, category, memo)

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

        return build_budget_flex(amount, category, period)

    # -----------------------------------------------------------------------
    # INTENT: ask_summary — ดูสรุปยอด (Premium Flex Message)
    # -----------------------------------------------------------------------
    elif intent == "ask_summary":
        try:
            summary = await get_monthly_summary(group_id)
        except Exception as exc:
            logger.error("Summary error: %s", exc)
            return "⚠️ ดึงข้อมูลสรุปไม่ได้ กรุณาลองใหม่อีกครั้งนะครับ"

        return build_summary_flex(summary)

    # -----------------------------------------------------------------------
    # INTENT: unknown
    # -----------------------------------------------------------------------
    else:
        return f"❓ วินัยไม่เข้าใจข้อความนะครับ — {error_msg or 'กรุณาลองพิมพ์ใหม่อีกครั้งนะครับ'}"


# ===========================================================================
# FASTAPI ENDPOINTS
# ===========================================================================


# ============================================================
# LIFF Dashboard — serve HTML from Railway (no Vercel needed)
# ============================================================
LIFF_HTML = """
<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FinCouple AI — Dashboard</title>
  <!-- Scripts loaded dynamically -->
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --pink: #FF6B9D;
      --purple: #C44BC4;
      --green: #4CAF50;
      --red: #F44336;
      --yellow: #FFC107;
      --bg: #0F0F1A;
      --card: #1A1A2E;
      --card2: #16213E;
      --text: #E0E0E0;
      --muted: #888;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Segoe UI', 'Noto Sans Thai', sans-serif;
      min-height: 100vh;
    }

    /* Header */
    .header {
      background: linear-gradient(135deg, var(--pink), var(--purple));
      padding: 20px 16px 28px;
      text-align: center;
      position: relative;
    }
    .header h1 { font-size: 1.3rem; font-weight: 700; letter-spacing: 0.5px; }
    .header .month { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
    .avatar { width: 48px; height: 48px; border-radius: 50%; border: 2px solid white; margin: 0 auto 10px; display: block; }

    /* Cards */
    .container { padding: 0 12px 24px; margin-top: -16px; }

    .summary-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      margin-bottom: 14px;
    }
    .summary-card {
      background: var(--card);
      border-radius: 14px;
      padding: 12px 8px;
      text-align: center;
    }
    .summary-card .label { font-size: 0.68rem; color: var(--muted); margin-bottom: 4px; }
    .summary-card .value { font-size: 1rem; font-weight: 700; }
    .summary-card .value.income { color: #4ADE80; }
    .summary-card .value.expense { color: #F87171; }
    .summary-card .value.balance-pos { color: #60A5FA; }
    .summary-card .value.balance-neg { color: #F87171; }

    .section { background: var(--card); border-radius: 16px; padding: 16px; margin-bottom: 14px; }
    .section-title { font-size: 0.9rem; font-weight: 600; margin-bottom: 14px; color: var(--pink); display: flex; align-items: center; gap: 6px; }

    /* Pie chart */
    .chart-wrap { position: relative; width: 180px; height: 180px; margin: 0 auto 12px; }

    /* Budget bars */
    .budget-item { margin-bottom: 12px; }
    .budget-header { display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 5px; }
    .budget-header .cat { color: var(--text); }
    .budget-header .amounts { color: var(--muted); }
    .bar-bg { background: #2a2a40; border-radius: 99px; height: 8px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
    .bar-fill.green { background: linear-gradient(90deg, #4ADE80, #22C55E); }
    .bar-fill.yellow { background: linear-gradient(90deg, #FCD34D, #F59E0B); }
    .bar-fill.red { background: linear-gradient(90deg, #F87171, #EF4444); }

    /* Transactions */
    .tx-item { display: flex; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid #2a2a40; }
    .tx-item:last-child { border-bottom: none; }
    .tx-emoji { width: 36px; height: 36px; border-radius: 10px; background: #2a2a40; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; flex-shrink: 0; }
    .tx-info { flex: 1; min-width: 0; }
    .tx-memo { font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .tx-cat { font-size: 0.72rem; color: var(--muted); margin-top: 2px; }
    .tx-amount { font-size: 0.95rem; font-weight: 700; flex-shrink: 0; }
    .tx-amount.expense { color: #F87171; }
    .tx-amount.income { color: #4ADE80; }

    /* Loading / Error */
    .loading { text-align: center; padding: 60px 20px; color: var(--muted); }
    .loading .spinner { width: 36px; height: 36px; border: 3px solid #333; border-top-color: var(--pink); border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 16px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .error-box { background: #2a1a1a; border: 1px solid #F87171; border-radius: 14px; padding: 20px; text-align: center; color: #F87171; margin: 20px 0; }

    .empty { text-align: center; color: var(--muted); padding: 20px 0; font-size: 0.85rem; }

    /* No group state */
    .no-group { text-align: center; padding: 40px 20px; }
    .no-group .emoji { font-size: 3rem; margin-bottom: 16px; }
    .no-group h2 { font-size: 1.1rem; margin-bottom: 8px; }
    .no-group p { color: var(--muted); font-size: 0.85rem; line-height: 1.6; }
  </style>
</head>
<body>

<div id="app">
  <div class="loading">
    <div class="spinner"></div>
    <p>กำลังโหลด...</p>
  </div>
</div>

<script>
// ============================================================
// CONFIG — แก้ค่านี้หลังได้ LIFF ID และ URL จาก Railway
// ============================================================
const LIFF_ID = "2010520479-6TrRjatU";    // LINE Login channel: FinCouple Dashboard
const API_BASE = "https://fincouple-bot-production.up.railway.app"; // URL จาก Railway

// ============================================================
// Category helpers
// ============================================================
const CAT_EMOJI = {
  food: "🍜", travel: "🚗", home: "🏠", shopping: "🛍️",
  entertainment: "🎬", savings: "💰", income: "💵", other: "📝"
};
const CAT_COLOR = {
  food: "#FF6B9D", travel: "#60A5FA", home: "#A78BFA", shopping: "#FCD34D",
  entertainment: "#F97316", savings: "#4ADE80", income: "#34D399", other: "#9CA3AF"
};

function fmt(n) { return Number(n).toLocaleString("th-TH", { maximumFractionDigits: 0 }); }

// ============================================================
// Render
// ============================================================
function renderDashboard(profile, data) {
  const balanceClass = data.balance >= 0 ? "balance-pos" : "balance-neg";
  const balanceSign = data.balance >= 0 ? "+" : "";

  // Build category chart data
  const cats = Object.entries(data.expense_by_category || {}).sort((a, b) => b[1] - a[1]);
  const chartLabels = cats.map(([k]) => `${CAT_EMOJI[k] || "📝"} ${k}`);
  const chartData = cats.map(([, v]) => v);
  const chartColors = cats.map(([k]) => CAT_COLOR[k] || "#9CA3AF");

  // Budget bars
  const budgets = data.budgets || {};
  const allCats = new Set([...Object.keys(data.expense_by_category || {}), ...Object.keys(budgets)]);
  const budgetHTML = [...allCats].filter(c => budgets[c]).map(cat => {
    const spent = data.expense_by_category?.[cat] || 0;
    const budget = budgets[cat] || 0;
    const pct = budget > 0 ? Math.min((spent / budget) * 100, 100) : 0;
    const cls = pct >= 100 ? "red" : pct >= 80 ? "yellow" : "green";
    return `
      <div class="budget-item">
        <div class="budget-header">
          <span class="cat">${CAT_EMOJI[cat] || "📝"} ${cat}</span>
          <span class="amounts">${fmt(spent)} / ${fmt(budget)} บ</span>
        </div>
        <div class="bar-bg"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
      </div>`;
  }).join("") || '<div class="empty">ยังไม่มีงบประมาณ<br>ตั้งงบได้ใน LINE เลยนะ 💡</div>';

  // Recent transactions
  const txHTML = (data.recent_transactions || []).slice(0, 10).map(tx => {
    const isExp = tx.type === "expense";
    const emoji = CAT_EMOJI[tx.category] || "📝";
    const sign = isExp ? "-" : "+";
    const cls = isExp ? "expense" : "income";
    const memo = tx.memo || tx.category || "-";
    return `
      <div class="tx-item">
        <div class="tx-emoji">${emoji}</div>
        <div class="tx-info">
          <div class="tx-memo">${memo}</div>
          <div class="tx-cat">${tx.category || "other"}</div>
        </div>
        <div class="tx-amount ${cls}">${sign}${fmt(tx.amount)}</div>
      </div>`;
  }).join("") || '<div class="empty">ยังไม่มีรายการ 🎉</div>';

  document.getElementById("app").innerHTML = `
    <div class="header">
      ${profile.pictureUrl ? `<img class="avatar" src="${profile.pictureUrl}" />` : ""}
      <h1>💕 FinCouple AI</h1>
      <div class="month">${data.month}</div>
    </div>

    <div class="container">
      <div class="summary-row">
        <div class="summary-card">
          <div class="label">รายรับ</div>
          <div class="value income">+${fmt(data.total_income)}</div>
        </div>
        <div class="summary-card">
          <div class="label">รายจ่าย</div>
          <div class="value expense">-${fmt(data.total_expense)}</div>
        </div>
        <div class="summary-card">
          <div class="label">คงเหลือ</div>
          <div class="value ${balanceClass}">${balanceSign}${fmt(data.balance)}</div>
        </div>
      </div>

      ${cats.length > 0 ? `
      <div class="section">
        <div class="section-title">📊 รายจ่ายแต่ละหมวด</div>
        <div class="chart-wrap"><canvas id="pieChart"></canvas></div>
      </div>` : ""}

      <div class="section">
        <div class="section-title">🎯 งบประมาณ</div>
        ${budgetHTML}
      </div>

      <div class="section">
        <div class="section-title">🕐 รายการล่าสุด</div>
        ${txHTML}
      </div>
    </div>
  `;

  // Draw pie chart
  if (cats.length > 0 && typeof Chart !== "undefined") {
    new Chart(document.getElementById("pieChart"), {
      type: "doughnut",
      data: { labels: chartLabels, datasets: [{ data: chartData, backgroundColor: chartColors, borderWidth: 0, hoverOffset: 8 }] },
      options: {
        cutout: "65%",
        plugins: {
          legend: { position: "bottom", labels: { color: "#ccc", font: { size: 11 }, padding: 10, boxWidth: 12 } }
        }
      }
    });
  }
}

function renderNoGroup(profile) {
  document.getElementById("app").innerHTML = `
    <div class="header">
      ${profile.pictureUrl ? `<img class="avatar" src="${profile.pictureUrl}" />` : ""}
      <h1>💕 FinCouple AI</h1>
    </div>
    <div class="container">
      <div class="no-group">
        <div class="emoji">👫</div>
        <h2>ยังไม่ได้เชื่อมกลุ่ม</h2>
        <p>กลับไปแชทกับบอทใน LINE<br>แล้วพิมพ์ <strong>/create</strong> เพื่อสร้างกลุ่ม<br>หรือ <strong>/join &lt;รหัส&gt;</strong> เพื่อเข้าร่วม</p>
      </div>
    </div>`;
}

function renderError(msg) {
  document.getElementById("app").innerHTML = `
    <div class="header"><h1>💕 FinCouple AI</h1></div>
    <div class="container"><div class="error-box">⚠️ ${msg}</div></div>`;
}

// ============================================================
// Main
// ============================================================
// ── Dynamic script loader ─────────────────────────
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error("Load failed: " + src));
    document.head.appendChild(s);
  });
}

async function main() {
  try {
    await loadScript("https://static.line-scdn.net/liff/edge/2/sdk.js");
  } catch (e1) {
    try {
      await loadScript("https://cdn.jsdelivr.net/npm/@line/liff@2.22.3/dist/liff.min.js");
    } catch (e2) {
      renderError("กรุณาเปิดผ่าน LINE app ครับ (SDK โหลดไม่ได้)");
      return;
    }
  }
  try {
    await loadScript("https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js");
  } catch (e) { /* chart optional */ }
  try {
    await liff.init({ liffId: LIFF_ID });
    if (!liff.isLoggedIn()) { liff.login(); return; }
    const profile = await liff.getProfile();
    const lineUserId = profile.userId;
    const res = await fetch(`${API_BASE}/api/summary?line_user_id=${encodeURIComponent(lineUserId)}`);
    if (res.status === 404) { renderNoGroup(profile); return; }
    if (!res.ok) throw new Error(`API error ${res.status}`);
    const data = await res.json();
    renderDashboard(profile, data);
  } catch (err) {
    console.error(err);
    renderError(err.message || "เกิดข้อผิดพลาด กรุณาลองใหม่");
  }
}
window.onload = main;
</script>
</body>
</html>
"""

@app.get("/liff", response_class=HTMLResponse)
async def liff_dashboard():
    return HTMLResponse(content=LIFF_HTML)

@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "วินัย Bot", "version": "5.0.0"})


@app.get("/api/summary")
async def api_summary(line_user_id: str = Query(...)) -> JSONResponse:
    try:
        user = await get_user(line_user_id)
    except Exception as exc:
        logger.error("api_summary get_user error: %s", exc)
        raise HTTPException(status_code=500, detail=f"DB error: {exc}")
    if not user or not user.get("group_id"):
        raise HTTPException(status_code=404, detail="User not found or not in a group")
    try:
        summary = await get_monthly_summary(user["group_id"])
    except Exception as exc:
        logger.error("API summary error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Summary error: {exc}")
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
        try:
            reply = await handle_text_event(event)
        except Exception as exc:
            logger.error("handle_text_event error: %s", exc, exc_info=True)
            reply = "⚠️ เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้งนะครับ"
        if isinstance(reply, dict):
            try:
                async with httpx.AsyncClient() as http:
                    r = await http.post(
                        "https://api.line.me/v2/bot/message/reply",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
                        },
                        json={
                            "replyToken": event.reply_token,
                            "messages": [{
                                "type": "flex",
                                "altText": reply["alt_text"],
                                "contents": reply["contents"],
                            }],
                        },
                        timeout=15.0,
                    )
                if not r.is_success:
                    logger.error("LINE Flex API error %s: %s", r.status_code, r.text)
            except Exception as exc:
                logger.error("Flex send error: %s", exc)
        else:
            try:
                out_msg = TextMessage(text=str(reply))
                with ApiClient(line_config) as api_client:
                    line_api = MessagingApi(api_client)
                    line_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[out_msg],
                        )
                    )
            except Exception as exc:
                logger.error("Text send error: %s", exc)
    return JSONResponse({"status": "ok"})
