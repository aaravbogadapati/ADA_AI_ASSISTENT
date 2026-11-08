"""
Planner tools for an AI agent.  Plain functions, JSON in / JSON out.

Setup:  pip install openpyxl
Use:    from planner_tools import TOOLS, SCHEMAS
        result = TOOLS["add_task"](title="Call dentist", date="2026-10-05")
Every function returns a dict that is safe to send straight back to the model.
"""
import functools
import json
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

from openpyxl import Workbook

# Folder that holds planner_tasks.json and planner_clock.json. The tools write there and the UI reads there.
# On Windows it uses your project folder; anywhere else (or if that folder is missing) it uses this script's folder.
DATA_DIR = Path(r"C:\My_Projects\Assistent_AI")
if not DATA_DIR.is_dir():
    DATA_DIR = Path(__file__).resolve().parent
DATA_FILE = DATA_DIR / "planner_tasks.json"
XLSX_FILE = DATA_DIR / "planner.xlsx"
DAY_START, DAY_END = 9 * 60, 17 * 60   # auto-plan window, in minutes (09:00-17:00)


# ---------- storage ----------
def _read_json(path, default):
    """Read a JSON file. Missing or empty -> default. Broken -> a clear error (never silently treated as empty,
    because the next save would overwrite your data). utf-8-sig also accepts files saved with a BOM."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return default
    if not text.strip():
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path.name} is not valid JSON ({e}). Fix or delete it; nothing was changed.")


def _load():
    data = _read_json(DATA_FILE, [])
    if not isinstance(data, list):
        raise ValueError(f"{DATA_FILE.name} must contain a list of tasks.")
    return data


def _save(tasks):
    DATA_FILE.write_text(json.dumps(tasks, indent=2), encoding="utf-8")


def _mins(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _hhmm(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def _check(value, fmt):
    datetime.strptime(value, fmt)   # raises ValueError if the format is wrong


# ---------- tools ----------
def add_task(title: str, date: str = "", time: str = "", minutes: int = 60,
             priority: int = 2, notes: str = "") -> dict:
    """Add a task. date=YYYY-MM-DD, time=HH:MM (both optional). priority: 1 low, 2 medium, 3 high."""
    try:
        if date:
            _check(date, "%Y-%m-%d")
        if time:
            _check(time, "%H:%M")
    except ValueError:
        return {"ok": False, "error": "date must be YYYY-MM-DD and time must be HH:MM"}
    tasks = _load()
    task = {"id": max((t["id"] for t in tasks), default=0) + 1, "title": title, "date": date,
            "time": time, "minutes": max(5, int(minutes)), "priority": int(priority),
            "notes": notes, "done": False}
    tasks.append(task)
    _save(tasks)
    return {"ok": True, "task": task}


def list_tasks(only_open: bool = False) -> dict:
    """List all tasks (set only_open=True to hide finished ones)."""
    tasks = [t for t in _load() if not (only_open and t["done"])]
    return {"ok": True, "tasks": tasks}


def complete_task(task_id: int) -> dict:
    """Mark a task as done."""
    tasks = _load()
    for t in tasks:
        if t["id"] == task_id:
            t["done"] = True
            _save(tasks)
            return {"ok": True, "task": t}
    return {"ok": False, "error": f"no task with id {task_id}"}


def delete_task(task_id: int) -> dict:
    """Delete a task."""
    tasks = _load()
    kept = [t for t in tasks if t["id"] != task_id]
    if len(kept) == len(tasks):
        return {"ok": False, "error": f"no task with id {task_id}"}
    _save(kept)
    return {"ok": True}


def get_schedule(day: str = "") -> dict:
    """Tasks for one day (YYYY-MM-DD), ordered by time. Defaults to today."""
    day = day or date.today().isoformat()
    items = sorted((t for t in _load() if t["date"] == day), key=lambda t: t["time"] or "99")
    return {"ok": True, "date": day, "tasks": items}


def get_calendar(year: int = 0, month: int = 0) -> dict:
    """Open-task count for every day of a month that has tasks. Defaults to this month."""
    today = date.today()
    prefix = f"{year or today.year:04d}-{month or today.month:02d}"
    days = {}
    for t in _load():
        if t["date"].startswith(prefix) and not t["done"]:
            days[t["date"]] = days.get(t["date"], 0) + 1
    return {"ok": True, "month": prefix, "tasks_per_day": dict(sorted(days.items()))}


def plan_tasks() -> dict:
    """Give every open task without a time a free slot (09:00-17:00), highest priority first."""
    tasks = _load()
    todo = sorted((t for t in tasks if not t["done"] and not t["time"]),
                  key=lambda t: (-t["priority"], t["date"] or "9999"))
    for t in todo:
        day = datetime.strptime(t["date"], "%Y-%m-%d").date() if t["date"] else date.today()
        for _ in range(120):
            busy = sorted((_mins(x["time"]), _mins(x["time"]) + x["minutes"]) for x in tasks
                          if x is not t and x["date"] == day.isoformat() and x["time"])
            start = DAY_START
            for s, e in busy:
                if start + t["minutes"] <= s:
                    break
                start = max(start, e)
            if start + t["minutes"] <= DAY_END:
                t["date"], t["time"] = day.isoformat(), _hhmm(start)
                break
            day += timedelta(days=1)
    _save(tasks)
    return {"ok": True, "planned": [{"id": t["id"], "title": t["title"], "date": t["date"], "time": t["time"]}
                                    for t in todo if t["time"]]}


def export_excel(path: str = "") -> dict:
    """Save all tasks to an Excel file (sheets: Tasks, Schedule). Returns the file path."""
    tasks = _load()
    out = Path(path) if path else XLSX_FILE
    head = ["ID", "Task", "Date", "Start", "End", "Minutes", "Priority", "Done", "Notes"]
    names = {1: "Low", 2: "Medium", 3: "High"}

    def row(t):
        end = _hhmm(_mins(t["time"]) + t["minutes"]) if t["time"] else ""
        return [t["id"], t["title"], t["date"], t["time"], end, t["minutes"],
                names.get(t["priority"], ""), "Yes" if t["done"] else "No", t["notes"]]

    wb = Workbook()
    ws = wb.active
    ws.title = "Tasks"
    ws.append(head)
    for t in tasks:
        ws.append(row(t))
    ws2 = wb.create_sheet("Schedule")
    ws2.append(head)
    for t in sorted((t for t in tasks if t["date"]), key=lambda t: (t["date"], t["time"] or "99")):
        ws2.append(row(t))
    wb.save(out)
    return {"ok": True, "file": str(out.resolve()), "task_count": len(tasks)}


# ---------- alarms, timers, stopwatches ----------
# Nothing blocks: alarms/timers are saved with a due time. Your agent loop calls
# check_alerts() now and then (or use start_watcher below) to see what has gone off.
CLOCK_FILE = DATA_DIR / "planner_clock.json"
_LOCK = threading.RLock()


def _locked(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        with _LOCK:
            return f(*args, **kwargs)
    return wrapper


def _load_clock():
    d = _read_json(CLOCK_FILE, {})
    if not isinstance(d, dict):
        raise ValueError(f"{CLOCK_FILE.name} must contain an object with alerts and stopwatches.")
    d.setdefault("alerts", [])
    d.setdefault("stopwatches", {})
    return d


def _save_clock(d):
    CLOCK_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def _now():
    return datetime.now().replace(microsecond=0)


def _nice(seconds):
    """Spoken-style duration ('4 minutes 41 seconds'). Small models misread raw seconds, so tools return this."""
    if seconds < 60:
        n = round(seconds, 1)
        n = int(n) if n == int(n) else n
        return f"{n} second" + ("" if n == 1 else "s")
    h, rest = divmod(round(seconds), 3600)
    m, sec = divmod(rest, 60)
    parts = []
    if h:
        parts.append(f"{h} hour" + ("" if h == 1 else "s"))
    if m:
        parts.append(f"{m} minute" + ("" if m == 1 else "s"))
    if sec and not h:
        parts.append(f"{sec} second" + ("" if sec == 1 else "s"))
    return " ".join(parts)


def _when(due):
    """'4:42 PM', 'tomorrow at 7:30 AM' or 'Monday 5 October at 9:00 AM'."""
    now, clock = _now(), due.strftime("%I:%M %p").lstrip("0")
    if due.date() == now.date():
        return clock
    if due.date() == now.date() + timedelta(days=1):
        return f"tomorrow at {clock}"
    return f"{due:%A} {due.day} {due:%B} at {clock}"


def _public(a):
    """What the model sees for an alarm/timer: friendly text, no raw numbers to misread."""
    due = datetime.fromisoformat(a["due"])
    left = max((due - _now()).total_seconds(), 0)
    return {"id": a["id"], "kind": a["kind"], "label": a["label"], "rings_in": _nice(left),
            "rings_at": _when(due), "repeat_daily": a["repeat_daily"]}


def _new_alert(kind, label, due, repeat_daily=False):
    d = _load_clock()
    alert = {"id": max((a["id"] for a in d["alerts"]), default=0) + 1, "kind": kind, "label": label,
             "due": due.isoformat(), "repeat_daily": repeat_daily}
    d["alerts"].append(alert)
    _save_clock(d)
    return alert


@_locked
def set_alarm(at: str, label: str = "Alarm", day: str = "", repeat_daily: bool = False) -> dict:
    """Alarm at a clock time (HH:MM, 24h). Without `day` it rings at the next occurrence of that time."""
    try:
        clock = datetime.strptime(at, "%H:%M").time()
        base = datetime.strptime(day, "%Y-%m-%d").date() if day else _now().date()
    except ValueError:
        return {"ok": False, "error": "at must be HH:MM and day must be YYYY-MM-DD"}
    due = datetime.combine(base, clock)
    if due <= _now():
        if day:
            return {"ok": False, "error": "that time is in the past"}
        due += timedelta(days=1)
    return {"ok": True, "alert": _public(_new_alert("alarm", label, due, repeat_daily))}


@_locked
def set_timer(hours: float = 0, minutes: float = 0, seconds: float = 0, label: str = "Timer") -> dict:
    """Countdown timer that goes off after the given duration."""
    total = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    if total <= 0:
        return {"ok": False, "error": "give a duration greater than zero"}
    due = (datetime.now() + timedelta(seconds=total)).replace(microsecond=0)
    return {"ok": True, "alert": _public(_new_alert("timer", label, due))}


@_locked
def list_alerts() -> dict:
    """All active alarms and timers, with how long until each one rings."""
    alerts = sorted(_load_clock()["alerts"], key=lambda a: a["due"])
    return {"ok": True, "alerts": [_public(a) for a in alerts]}


@_locked
def cancel_alert(alert_id: int) -> dict:
    """Cancel an alarm or timer."""
    d = _load_clock()
    kept = [a for a in d["alerts"] if a["id"] != alert_id]
    if len(kept) == len(d["alerts"]):
        return {"ok": False, "error": f"no alarm or timer with id {alert_id}"}
    d["alerts"] = kept
    _save_clock(d)
    return {"ok": True}


@_locked
def cancel_all_alerts(kind: str = "") -> dict:
    """Cancel every alarm and timer, or only one kind (kind='alarm' or 'timer')."""
    if kind not in ("", "alarm", "timer"):
        return {"ok": False, "error": "kind must be 'alarm', 'timer', or empty for both"}
    d = _load_clock()
    kept = [a for a in d["alerts"] if kind and a["kind"] != kind]
    cancelled = len(d["alerts"]) - len(kept)
    d["alerts"] = kept
    _save_clock(d)
    return {"ok": True, "cancelled": cancelled}


@_locked
def check_alerts() -> dict:
    """Return alarms/timers that have gone off since the last check (each is reported once)."""
    d, now = _load_clock(), _now()
    fired, keep = [], []
    for a in d["alerts"]:
        due = datetime.fromisoformat(a["due"])
        if due > now:
            keep.append(a)
            continue
        fired.append(dict(a))
        if a["repeat_daily"]:                 # daily alarms re-arm for the next day
            while due <= now:
                due += timedelta(days=1)
            a["due"] = due.isoformat()
            keep.append(a)
    if fired:                                 # only touch the disk when something actually went off
        d["alerts"] = keep
        _save_clock(d)
    return {"ok": True, "fired": fired}


def start_watcher(on_fire, every: float = 1.0):
    """Python helper (not an agent tool). Checks alerts in the background and calls on_fire(alert).
    Returns a function; call it to stop the watcher."""
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                for alert in check_alerts()["fired"]:
                    on_fire(alert)
            except Exception as e:                # keep watching even if the file is briefly unreadable
                print("Alert watcher:", e)
            stop.wait(every)

    threading.Thread(target=loop, daemon=True).start()
    return stop.set


def _elapsed(sw):
    extra = (datetime.now() - datetime.fromisoformat(sw["started"])).total_seconds() if sw["running"] else 0
    return round(sw["elapsed"] + extra, 2)


def _view(name, sw):
    laps = [{"lap": l["lap"], "lap_time": _nice(l["lap_seconds"]), "total_time": _nice(l["total_seconds"])}
            for l in sw["laps"]]
    return {"name": name, "running": sw["running"], "elapsed": _nice(_elapsed(sw)), "laps": laps}


@_locked
def stopwatch_start(name: str = "default") -> dict:
    """Start a stopwatch (or resume it if it was stopped)."""
    d = _load_clock()
    sw = d["stopwatches"].setdefault(name, {"running": False, "elapsed": 0.0, "started": "", "laps": []})
    if not sw["running"]:
        sw["running"], sw["started"] = True, datetime.now().isoformat()
    _save_clock(d)
    return {"ok": True, **_view(name, sw)}


@_locked
def stopwatch_lap(name: str = "default") -> dict:
    """Record a lap on a running stopwatch."""
    d = _load_clock()
    sw = d["stopwatches"].get(name)
    if not sw or not sw["running"]:
        return {"ok": False, "error": f"stopwatch '{name}' is not running"}
    total = _elapsed(sw)
    prev = sw["laps"][-1]["total_seconds"] if sw["laps"] else 0
    sw["laps"].append({"lap": len(sw["laps"]) + 1, "lap_seconds": round(total - prev, 2), "total_seconds": total})
    _save_clock(d)
    return {"ok": True, **_view(name, sw)}


@_locked
def stopwatch_stop(name: str = "default") -> dict:
    """Pause a stopwatch and keep its time (start it again to resume)."""
    d = _load_clock()
    sw = d["stopwatches"].get(name)
    if not sw:
        return {"ok": False, "error": f"no stopwatch named '{name}'"}
    if sw["running"]:
        sw["elapsed"], sw["running"] = _elapsed(sw), False
        _save_clock(d)
    return {"ok": True, **_view(name, sw)}


@_locked
def stopwatch_status(name: str = "") -> dict:
    """Elapsed time and laps for one stopwatch, or for all of them if name is empty."""
    sws = _load_clock()["stopwatches"]
    if not name:
        return {"ok": True, "stopwatches": [_view(n, s) for n, s in sws.items()]}
    if name not in sws:
        return {"ok": False, "error": f"no stopwatch named '{name}'"}
    return {"ok": True, **_view(name, sws[name])}


@_locked
def stopwatch_reset(name: str = "default") -> dict:
    """Delete a stopwatch (time and laps)."""
    d = _load_clock()
    if d["stopwatches"].pop(name, None) is None:
        return {"ok": False, "error": f"no stopwatch named '{name}'"}
    _save_clock(d)
    return {"ok": True}


def snapshot() -> dict:
    """Everything in planner_tasks.json and planner_clock.json, for the UI (not an agent tool)."""
    try:
        return _snapshot()
    except ValueError as e:                       # a JSON file is broken: tell the UI why
        return {"ok": False, "error": str(e)}


def _snapshot():
    with _LOCK:
        now, clock, alerts = _now(), _load_clock(), []
        for a in sorted(clock["alerts"], key=lambda a: a["due"]):
            due = datetime.fromisoformat(a["due"])
            alerts.append({"id": a["id"], "kind": a["kind"], "label": a["label"], "due": a["due"],
                           "seconds_left": max(int((due - now).total_seconds()), 0),
                           "rings_at": _when(due), "repeat_daily": a["repeat_daily"]})
        stopwatches = [{"name": n, "running": sw["running"], "elapsed": _elapsed(sw), "laps": sw["laps"]}
                       for n, sw in clock["stopwatches"].items()]
        return {"ok": True, "tasks": _load(), "alerts": alerts, "stopwatches": stopwatches}


# ---------- hand these to your agent ----------
TOOLS = {f.__name__: f for f in
         (add_task, list_tasks, complete_task, delete_task, get_schedule, get_calendar, plan_tasks, export_excel,
          set_alarm, set_timer, list_alerts, cancel_alert, cancel_all_alerts, check_alerts,
          stopwatch_start, stopwatch_lap, stopwatch_stop, stopwatch_status, stopwatch_reset)}


def _schema(name, desc, props=None, required=None):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props or {}, "required": required or []}}


S, I, B, N = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}, {"type": "number"}
SCHEMAS = [   # Anthropic tool-use format; for OpenAI wrap each as {"type":"function","function":{... "parameters": input_schema}}
    _schema("add_task", "Add a task. date is YYYY-MM-DD, time is HH:MM, priority 1 low to 3 high.",
            {"title": S, "date": S, "time": S, "minutes": I, "priority": I, "notes": S}, ["title"]),
    _schema("list_tasks", "List tasks.", {"only_open": B}),
    _schema("complete_task", "Mark a task done.", {"task_id": I}, ["task_id"]),
    _schema("delete_task", "Delete a task.", {"task_id": I}, ["task_id"]),
    _schema("get_schedule", "Get one day's tasks in time order (defaults to today).", {"day": S}),
    _schema("get_calendar", "Get open-task counts per day for a month.", {"year": I, "month": I}),
    _schema("plan_tasks", "Auto-assign free time slots to open tasks that have no time."),
    _schema("export_excel", "Save all tasks to an Excel file.", {"path": S}),
    _schema("set_alarm", "Set an alarm for a clock time (HH:MM, 24h). Optional day YYYY-MM-DD; repeat_daily repeats it.",
            {"at": S, "label": S, "day": S, "repeat_daily": B}, ["at"]),
    _schema("set_timer", "Set a countdown timer. Give hours, minutes and/or seconds.",
            {"hours": N, "minutes": N, "seconds": N, "label": S}),
    _schema("list_alerts", "List active alarms and timers and when each rings."),
    _schema("cancel_alert", "Cancel an alarm or timer.", {"alert_id": I}, ["alert_id"]),
    _schema("cancel_all_alerts", "Cancel ALL alarms and timers at once. Optional kind: alarm or timer.", {"kind": S}),
    _schema("check_alerts", "Get alarms/timers that have gone off since the last check. Call this regularly."),
    _schema("stopwatch_start", "Start or resume a named stopwatch.", {"name": S}),
    _schema("stopwatch_lap", "Record a lap on a running stopwatch.", {"name": S}),
    _schema("stopwatch_stop", "Pause a stopwatch, keeping its time.", {"name": S}),
    _schema("stopwatch_status", "Elapsed time and laps for one stopwatch, or all if name is empty.", {"name": S}),
    _schema("stopwatch_reset", "Delete a stopwatch.", {"name": S}),
]


def run_tool(name: str, args: dict) -> str:
    """Call a tool by name with a dict of arguments; returns a JSON string for the model."""
    try:
        return json.dumps(TOOLS[name](**args))
    except Exception as e:   # never crash the agent loop
        return json.dumps({"ok": False, "error": str(e)})