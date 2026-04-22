import os
import io
import base64
import asyncio
import logging
import time
from dotenv import load_dotenv

import anthropic
from openai import AsyncOpenAI
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardMarkup, KeyboardButton
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
from telegram.constants import ChatAction, ParseMode

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Models ────────────────────────────────────────────────────────────────────

CLAUDE_MODELS = {
    "sonnet": ("claude-sonnet-4-6", "Sonnet 4.6  —  быстрый"),
    "opus":   ("claude-opus-4-7",   "Opus 4.7  —  мощный"),
}
GPT_MODELS = {
    "4o":  ("gpt-4o",   "GPT-4o  —  быстрый"),
    "41":  ("gpt-4.1",  "GPT-4.1  —  умный"),
    "o3":  ("o3",       "o3  —  рассуждение"),
}

PROVIDERS = {
    "claude": {"label": "🟠 Claude", "models": CLAUDE_MODELS, "default_model": "sonnet"},
    "gpt":    {"label": "🟢 GPT",    "models": GPT_MODELS,    "default_model": "4o"},
}

SYSTEM_PROMPT = (
    "Ты SAO — умный AI-ассистент. Отвечай естественно, как в переписке. "
    "Используй Markdown только когда реально нужно (код, списки). "
    "Говори на языке пользователя."
)

# ── State ─────────────────────────────────────────────────────────────────────
# chat: {"name": str, "msgs": list, "model_label": str, "tokens": int}

users: dict[int, dict] = {}

def _new_provider_state(model_key: str) -> dict:
    return {
        "model": model_key,
        "active": 0,
        "chats": [{"name": "Новый чат", "msgs": [], "model_label": "", "tokens": 0}],
        "tokens_total": 0,
        "tokens_remaining": None,  # from rate limit headers
    }

def get_user(uid: int) -> dict:
    if uid not in users:
        users[uid] = {
            "provider": "claude",
            "claude": _new_provider_state("sonnet"),
            "gpt":    _new_provider_state("4o"),
        }
    return users[uid]


def current_chat(u: dict) -> dict:
    p = u["provider"]
    return u[p]["chats"][u[p]["active"]]


def current_model_id(u: dict) -> str:
    p = u["provider"]
    model_key = u[p]["model"]
    return PROVIDERS[p]["models"][model_key][0]


def current_model_label(u: dict) -> str:
    p = u["provider"]
    model_key = u[p]["model"]
    return PROVIDERS[p]["models"][model_key][1]


# ── Keyboards ─────────────────────────────────────────────────────────────────

MENU_KB = ReplyKeyboardMarkup([[KeyboardButton("☰  Меню")]], resize_keyboard=True)


def main_menu_kb(u: dict) -> InlineKeyboardMarkup:
    p = u["provider"]
    rows = []
    # Provider tabs
    tabs = []
    for pid, pdata in PROVIDERS.items():
        mark = "●  " if pid == p else ""
        tabs.append(InlineKeyboardButton(f"{mark}{pdata['label']}", callback_data=f"tab:{pid}"))
    rows.append(tabs)
    # Model selector
    models = PROVIDERS[p]["models"]
    cur_model = u[p]["model"]
    model_btns = []
    for mk, (_, mlabel) in models.items():
        mark = "✓ " if mk == cur_model else ""
        short = mlabel.split("  ")[0]
        model_btns.append(InlineKeyboardButton(f"{mark}{short}", callback_data=f"model:{mk}"))
    rows.append(model_btns)
    # Chats list
    chats = u[p]["chats"]
    active = u[p]["active"]
    for i, chat in enumerate(chats):
        mark = "▶ " if i == active else "    "
        name = chat["name"][:22]
        model_tag = f"  [{chat['model_label']}]" if chat.get("model_label") else ""
        rows.append([InlineKeyboardButton(f"{mark}{name}{model_tag}", callback_data=f"chat:{i}")])
    # New chat
    rows.append([InlineKeyboardButton("＋  Новый чат", callback_data="newchat")])
    return InlineKeyboardMarkup(rows)


def back_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("☰  Меню", callback_data="menu")]])


# ── Menu message ──────────────────────────────────────────────────────────────

def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1000:
        return f"{n/1000:.1f}K"
    return str(n)

def menu_text(u: dict) -> str:
    lines = []
    for pid, pdata in PROVIDERS.items():
        ps = u[pid]
        mark = ">>  " if pid == u["provider"] else "    "
        rem = ps.get("tokens_remaining")
        if rem is not None:
            tok_str = f"{_fmt_tokens(rem)} ост."
        else:
            tok_str = "—"
        lines.append(f"{mark}{pdata['label']}  —  {tok_str}")
    lines.append("")
    p = u["provider"]
    chat = current_chat(u)
    model = current_model_label(u)
    msgs = len(chat["msgs"]) // 2
    lines.append(f"Модель: {model}")
    lines.append(f"Чат: {chat['name']}  ({msgs} сообщ.)")
    return "\n".join(lines)


# ── Handlers ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = get_user(update.effective_user.id)
    name = update.effective_user.first_name or ""
    await update.message.reply_text(
        f"Привет{', ' + name if name else ''}. Я SAO.\nПросто пиши — я отвечу.",
        reply_markup=MENU_KB,
    )


async def handle_menu_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = get_user(update.effective_user.id)
    await update.message.reply_text(menu_text(u), reply_markup=main_menu_kb(u))


async def callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    u = get_user(q.from_user.id)
    d = q.data

    if d == "menu":
        await q.edit_message_text(menu_text(u), reply_markup=main_menu_kb(u))

    elif d.startswith("tab:"):
        u["provider"] = d.split(":")[1]
        await q.edit_message_text(menu_text(u), reply_markup=main_menu_kb(u))

    elif d.startswith("model:"):
        mk = d.split(":")[1]
        p = u["provider"]
        if mk in PROVIDERS[p]["models"]:
            u[p]["model"] = mk
        await q.edit_message_text(menu_text(u), reply_markup=main_menu_kb(u))

    elif d.startswith("chat:"):
        idx = int(d.split(":")[1])
        p = u["provider"]
        u[p]["active"] = idx
        chat = current_chat(u)
        await q.edit_message_text(
            f"Чат: {chat['name']}\n{len(chat['msgs']) // 2} сообщений",
            reply_markup=back_kb(),
        )

    elif d == "newchat":
        p = u["provider"]
        u[p]["chats"].append({"name": "Новый чат", "msgs": [], "model_label": "", "tokens": 0})
        u[p]["active"] = len(u[p]["chats"]) - 1
        await q.edit_message_text("Новый чат создан. Напиши что-нибудь.", reply_markup=back_kb())


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.message.text == "☰  Меню":
        await handle_menu_button(update, ctx)
        return
    u = get_user(update.effective_user.id)
    chat = current_chat(u)
    chat["msgs"].append({"role": "user", "content": update.message.text})
    # Auto-name chat from first message
    if len(chat["msgs"]) == 1:
        chat["name"] = update.message.text[:30]
    await _reply(update, ctx, u)


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = get_user(update.effective_user.id)
    caption = update.message.caption or "Что на этом изображении?"
    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    photo = update.message.photo[-1]
    file = await ctx.bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    p = u["provider"]
    chat = current_chat(u)

    if p == "claude":
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
            {"type": "text", "text": caption},
        ]
    else:
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": caption},
        ]

    chat["msgs"].append({"role": "user", "content": content})
    if len(chat["msgs"]) == 1:
        chat["name"] = "Фото"
    await _reply(update, ctx, u)


async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = get_user(update.effective_user.id)
    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    voice = update.message.voice or update.message.audio
    file = await ctx.bot.get_file(voice.file_id)
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    buf.name = "voice.ogg"
    buf.seek(0)

    try:
        transcript = await openai_client.audio.transcriptions.create(
            model="whisper-1", file=buf, response_format="text"
        )
    except Exception as e:
        await update.message.reply_text(f"Не смог расшифровать: {e}")
        return

    await update.message.reply_text(f"_{transcript}_", parse_mode=ParseMode.MARKDOWN)

    chat = current_chat(u)
    chat["msgs"].append({"role": "user", "content": transcript})
    if len(chat["msgs"]) == 1:
        chat["name"] = transcript[:30]
    await _reply(update, ctx, u)


async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = get_user(update.effective_user.id)
    doc = update.message.document
    caption = update.message.caption or "Проанализируй этот файл."
    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    file = await ctx.bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    buf.seek(0)

    fname = doc.file_name or ""
    text_content = None

    try:
        if fname.endswith(".pdf") and PDF_SUPPORT:
            reader = PyPDF2.PdfReader(buf)
            text_content = "\n\n".join(
                p.extract_text() for p in reader.pages if p.extract_text()
            )[:15000]
        elif fname.endswith(".docx") and DOCX_SUPPORT:
            docx = DocxDocument(buf)
            text_content = "\n".join(p.text for p in docx.paragraphs)[:15000]
        elif fname.endswith((".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".html", ".css", ".xml")):
            text_content = buf.read().decode("utf-8", errors="ignore")[:15000]
        else:
            await update.message.reply_text("Формат не поддерживается. Пришли PDF, TXT, DOCX или код-файл.")
            return
    except Exception as e:
        await update.message.reply_text(f"Не смог прочитать файл: {e}")
        return

    chat = current_chat(u)
    prompt = f"{caption}\n\n```\n{text_content}\n```"
    chat["msgs"].append({"role": "user", "content": prompt})
    if len(chat["msgs"]) == 1:
        chat["name"] = fname[:30]
    await _reply(update, ctx, u)


# ── AI calls ──────────────────────────────────────────────────────────────────

async def _reply(update: Update, ctx: ContextTypes.DEFAULT_TYPE, u: dict):
    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    p = u["provider"]
    chat = current_chat(u)
    msgs = chat["msgs"]
    is_first_exchange = len(msgs) == 1  # user msg added, no assistant yet

    try:
        if p == "claude":
            answer, tokens, remaining = await _ask_claude(msgs, current_model_id(u))
        else:
            answer, tokens, remaining = await _ask_gpt(msgs, current_model_id(u))
    except Exception as e:
        logger.error(e)
        err = str(e)
        if "insufficient_quota" in err or "credit balance is too low" in err or "quota" in err.lower():
            provider_name = "OpenAI (platform.openai.com/settings/billing)" if p == "gpt" else "Anthropic (console.anthropic.com/settings/billing)"
            other = "🟠 Claude" if p == "gpt" else "🟢 GPT"
            await update.message.reply_text(
                f"Баланс API исчерпан.\n\nПополни: {provider_name}\n\nВажно: подписка ChatGPT Plus / Claude Pro не даёт доступ к API — нужно отдельно пополнить баланс на платформе разработчика.\n\nПока переключись на {other}."
            )
        elif "rate_limit" in err.lower():
            await update.message.reply_text("Слишком много запросов, подожди немного и повтори.")
        else:
            await update.message.reply_text(f"Ошибка: {e}")
        return

    # track tokens
    chat["tokens"] = chat.get("tokens", 0) + tokens
    u[p]["tokens_total"] = u[p].get("tokens_total", 0) + tokens
    if remaining is not None:
        u[p]["tokens_remaining"] = remaining

    chat["msgs"].append({"role": "assistant", "content": answer})
    chat["model_label"] = u[p]["model"]  # store model key used

    # keep last 30 messages
    if len(chat["msgs"]) > 30:
        chat["msgs"] = chat["msgs"][-30:]

    # generate smart chat name after first exchange
    if is_first_exchange:
        asyncio.create_task(_name_chat(chat, msgs[0]["content"] if isinstance(msgs[0]["content"], str) else "Медиа"))

    # send answer
    if len(answer) > 4000:
        for i in range(0, len(answer), 4000):
            try:
                await update.message.reply_text(answer[i:i+4000], parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(answer[i:i+4000])
    else:
        try:
            await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(answer)


async def _name_chat(chat: dict, first_msg: str):
    """Generate a short smart name for the chat using GPT-4o-mini."""
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Придумай очень короткое название (2-4 слова) для чата на основе первого сообщения пользователя. Только название, без кавычек и точек."},
                {"role": "user", "content": first_msg[:300]},
            ],
            max_completion_tokens=20,
        )
        name = resp.choices[0].message.content.strip()[:30]
        if name:
            chat["name"] = name
    except Exception:
        pass


async def _ask_claude(history: list, model: str) -> tuple[str, int, int | None]:
    messages = _normalize_for_claude(history)
    loop = asyncio.get_event_loop()

    def _call():
        return anthropic_client.messages.with_raw_response.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

    raw = await loop.run_in_executor(None, _call)
    resp = raw.parse()
    tokens = resp.usage.input_tokens + resp.usage.output_tokens
    remaining = raw.headers.get("anthropic-ratelimit-tokens-remaining")
    remaining = int(remaining) if remaining else None
    return resp.content[0].text, tokens, remaining


async def _ask_gpt(history: list, model: str) -> tuple[str, int, int | None]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += _normalize_for_gpt(history)
    if model == "o3":
        messages = [m for m in messages if m["role"] != "system"]
        messages.insert(0, {"role": "user", "content": SYSTEM_PROMPT + "\n\n---\n\nТеперь отвечай на сообщения пользователя."})
    raw = await openai_client.chat.completions.with_raw_response.create(
        model=model,
        messages=messages,
        max_completion_tokens=4096,
    )
    resp = raw.parse()
    tokens = resp.usage.prompt_tokens + resp.usage.completion_tokens
    remaining = raw.headers.get("x-ratelimit-remaining-tokens")
    remaining = int(remaining) if remaining else None
    return resp.choices[0].message.content, tokens, remaining


def _normalize_for_claude(history: list) -> list:
    result = []
    for m in history:
        if isinstance(m["content"], list):
            parts = []
            for part in m["content"]:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        parts.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
                else:
                    parts.append(part)
            result.append({"role": m["role"], "content": parts})
        else:
            result.append(m)
    return result


def _normalize_for_gpt(history: list) -> list:
    result = []
    for m in history:
        if isinstance(m["content"], list):
            parts = []
            for part in m["content"]:
                if part.get("type") == "image":
                    src = part["source"]
                    parts.append({"type": "image_url", "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}})
                else:
                    parts.append(part)
            result.append({"role": m["role"], "content": parts})
        else:
            result.append(m)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(callback))
    app.add_handler(MessageHandler(filters.Regex(r"^☰"), handle_menu_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    print("SAO started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
