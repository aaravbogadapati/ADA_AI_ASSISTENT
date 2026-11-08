"""
ada_news.py - glue between news_get.py and Ada.

Does three jobs:
  1. NEWS_TOOLS      -> tool definitions in Ollama format (sent to the LLM)
  2. run_news_tool() -> runs a news tool call and returns a compact JSON string
                        sized for a small local model
  3. news_for_ui()   -> data for the News tab (/news endpoint in ada.py)

Must sit in the same folder as ada.py, news_get.py and feeds.txt.
"""

import json
from datetime import datetime, timezone
from urllib.parse import parse_qs

import news_get

# ---- tuning ----
GENERIC_ONLY = False   # True = give the LLM only get_news / search_news / get_all_news /
                       # list_categories (saves context; then drop the topic list from the Modelfile)
DEFAULT_LIMIT = 4      # articles per call when the model doesn't say
MAX_ARTICLES = 6       # hard cap per call (spoken news should be short)
ALL_PER_CATEGORY = 2   # get_all_news: articles per category
SUMMARY_CHARS = 180    # summary length sent to the LLM
UI_DEFAULT_LIMIT = 40  # articles shown in the News tab
UI_MAX_LIMIT = 80

_GENERIC = {"get_news", "search_news", "get_all_news", "list_categories"}


# --------------------------------------------------------------------------- #
# 1. Tool definitions (Ollama / OpenAI style, converted from news_get.TOOL_SCHEMAS)
# --------------------------------------------------------------------------- #
def _tool_defs():
    return [
        {"type": "function",
         "function": {"name": s["name"], "description": s["description"],
                      "parameters": s["input_schema"]}}
        for s in news_get.TOOL_SCHEMAS
        if not GENERIC_ONLY or s["name"] in _GENERIC
    ]


NEWS_TOOLS = _tool_defs()
_ALLOWED_ARGS = {s["name"]: set(s["input_schema"].get("properties", {}))
                 for s in news_get.TOOL_SCHEMAS}


def is_news_tool(name):
    return name in news_get.TOOL_FUNCTIONS


# --------------------------------------------------------------------------- #
# 2. Running a tool call for the LLM
# --------------------------------------------------------------------------- #
def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _age(iso):
    """'2 hours ago' - small models read this far better than a raw timestamp."""
    try:
        secs = (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    except (TypeError, ValueError):
        return ""
    mins = int(secs // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return _plural(mins, "minute") + " ago"
    hours = mins // 60
    if hours < 24:
        return _plural(hours, "hour") + " ago"
    return _plural(hours // 24, "day") + " ago"


def _brief(article, with_summary=True):
    out = {"title": article["title"], "source": article["source"],
           "age": _age(article["published"])}
    if with_summary and article["summary"]:
        out["summary"] = article["summary"][:SUMMARY_CHARS]
    return out  # links are left out on purpose: saves tokens and Ada never reads them aloud


def _focus_for(name, args):
    """What the News tab should show after this tool ran: (category, search text)."""
    if name == "get_news":
        return str(args.get("category", "all")).lower(), ""
    if name == "search_news":
        kw = args.get("keywords") or ""
        return "all", ",".join(kw) if isinstance(kw, list) else str(kw)
    if name == "get_all_news":
        return "all", ""
    if name.startswith("get_") and name.endswith("_news"):
        return name[4:-5], ""
    return None


def run_news_tool(name, args, on_focus=None):
    """Run one news tool. Always returns a JSON string with an 'ok' field."""
    fn = news_get.TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"ok": False, "error": f"Unknown tool '{name}'"})

    # drop arguments the model invented, coerce numbers it sent as strings
    clean = {k: v for k, v in (args or {}).items() if k in _ALLOWED_ARGS.get(name, set())}
    for key, cast in (("limit", int), ("limit_per_category", int), ("since_hours", float)):
        if key in clean:
            try:
                clean[key] = cast(clean[key])
            except (TypeError, ValueError):
                del clean[key]
    if clean.get("since_hours") is not None and clean["since_hours"] <= 0:
        del clean["since_hours"]

    if name == "get_all_news":
        n = clean.get("limit_per_category", ALL_PER_CATEGORY)
        clean["limit_per_category"] = max(1, min(ALL_PER_CATEGORY, n))
    elif name != "list_categories":
        clean["limit"] = max(1, min(MAX_ARTICLES, clean.get("limit", DEFAULT_LIMIT)))

    try:
        result = fn(**clean)
    except Exception as exc:
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    if name == "list_categories":
        payload = {"ok": True, **result}
    elif name == "get_all_news":
        payload = {"ok": True,
                   "briefing": {c: [_brief(a, False) for a in arts] for c, arts in result.items()}}
    else:
        errors = [e[:140] for e in result["errors"][:3]]
        if result["count"] == 0 and errors:
            payload = {"ok": False, "error": errors[0], "failed_feeds": len(result["errors"])}
        else:
            payload = {"ok": True, "category": result["category"], "count": result["count"],
                       "articles": [_brief(a) for a in result["articles"]]}
            if errors:
                payload["failed_feeds"] = len(result["errors"])

    if on_focus and payload.get("ok"):
        focus = _focus_for(name, clean)
        if focus:
            on_focus(*focus)
    return json.dumps(payload, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 3. Data for the News tab
# --------------------------------------------------------------------------- #
def categories_for_ui():
    try:
        return {"ok": True, **news_get.list_categories()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def news_for_ui(query_string):
    """query_string like 'cat=ai&q=openai&limit=40' -> JSON-able dict."""
    qs = parse_qs(query_string or "")
    cat = (qs.get("cat", ["all"])[0] or "all").strip().lower()
    text = qs.get("q", [""])[0].strip()
    try:
        limit = max(1, min(UI_MAX_LIMIT, int(qs.get("limit", [UI_DEFAULT_LIMIT])[0])))
    except ValueError:
        limit = UI_DEFAULT_LIMIT
    keywords = [k.strip() for k in text.split(",") if k.strip()] or None

    try:
        if cat == "all":
            result = news_get.search_news(keywords, limit=limit)
        else:
            result = news_get.get_news(cat, limit=limit, keywords=keywords)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    result["ok"] = True
    result["fetched"] = datetime.now(timezone.utc).isoformat()
    return result