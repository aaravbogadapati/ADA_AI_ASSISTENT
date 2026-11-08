import time
import threading
import queue
import numpy as np
import wave
import io
import sounddevice as sd
import requests
import re
from collections import deque
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import urlparse
from piper import PiperVoice
from faster_whisper import WhisperModel

# NEW: planner tools (tasks, schedule, calendar, alarms, timers, stopwatches).
# planner_tools.py must sit in the same folder as this file.
from schedualAGT import SCHEMAS, run_tool, start_watcher, snapshot

# NEWS: news tools. ada_news.py, news_get.py and feeds.txt must sit in this folder too.
import ada_news

# GESTURES: hand-gesture / room tracker. gesture_engine.py and hand_gesture_tracker_v19.py
# must sit in this folder too.
from gesture_engine import GestureEngine

gesture = None                 # GESTURES: created in __main__
gesture_cmds = queue.Queue()   # GESTURES: gesture thread -> main loop ("toggle" / "sleep" / "shutdown")

# ---- LLM setup ----
OLLAMA_MODEL = "ada6_lite"  # custom model built from your Modelfile
OLLAMA_NUM_CTX = 10000# NEWS: was 8192. Planner + news tool definitions and article text need the room.
MAX_TOOL_ROUNDS = 5    # NEW: stops a runaway tool loop
MAX_HISTORY = 30       # NEW: keeps the chat history inside the context window

# NEW: this is what makes Ada "see" the tools. They are sent with every request.
# Ollama wants OpenAI-style definitions, so convert from planner_tools.SCHEMAS.
# check_alerts is left out because the background watcher below handles alerts.
TOOLS = [
    {"type": "function",
     "function": {"name": s["name"], "description": s["description"], "parameters": s["input_schema"]}}
    for s in SCHEMAS if s["name"] != "check_alerts"
]
# NEWS: add the news tools (set GENERIC_ONLY = True in ada_news.py to send fewer of them)
TOOLS += ada_news.NEWS_TOOLS

# ---- STT setup (faster-whisper) ----
MODEL_SIZE = "base.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024               # samples per audio callback chunk
SILENCE_THRESHOLD = 500         # amplitude below this counts as silence - tune if it cuts you off or waits too long
SILENCE_DURATION = 0.8          # seconds of continuous silence that ends the recording
MAX_RECORD_SECONDS = 15         # hard cap so a stuck mic can't record forever
PRE_SPEECH_TIMEOUT = 10         # seconds to wait for you to start talking before giving up

print(f"Loading Whisper model '{MODEL_SIZE}' (device={DEVICE}, compute_type={COMPUTE_TYPE})...")
stt_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Whisper model loaded.")

SETTLE_TIME = 0.4  # seconds

ONSET_BLOCKS = 4  # consecutive loud blocks (~0.25s) needed before it counts as speech.
                  # Raise it if clicks/noise still trigger gold; lower it if short words get missed.

# ---- UI state server (serves ada_ui.html + /state JSON) ----
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UI_PORT = 8765
UI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ada_ui.html")
# mode: asleep | listening | user | thinking | ada
# NEWS: news_focus changes whenever Ada fetches news, so the News tab can show the same stories.
# GESTURES: tab_cmd changes whenever a swipe gesture asks the UI to change tab (dir: +1 next, -1 previous).
ui_state = {"mode": "asleep", "awake": False, "messages": [],
            "news_focus": {"id": 0, "cat": "", "q": ""},
            "tab_cmd": {"id": 0, "dir": 0}}
_msg_counter = {"n": 0}
_news_counter = {"n": 0}
_tab_counter = {"n": 0}

def set_state(**kw):
    ui_state.update(kw)

def add_message(role, text):
    text = (text or "").strip()
    if not text:
        return
    _msg_counter["n"] += 1
    ui_state["messages"].append({"id": _msg_counter["n"], "role": role, "text": text})
    del ui_state["messages"][:-100]  # keep the last 100

def set_idle():
    set_state(mode="listening" if ui_state["awake"] else "asleep")

# NEWS: called by ada_news after a successful news tool call
def news_focus(cat, q):
    _news_counter["n"] += 1
    set_state(news_focus={"id": _news_counter["n"], "cat": cat, "q": q})

# GESTURES: called by the gesture engine on a swipe (+1 = next tab, -1 = previous tab)
def tab_move(direction):
    _tab_counter["n"] += 1
    set_state(tab_cmd={"id": _tab_counter["n"], "dir": direction})

class UIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/camera.mjpg":  # GESTURES: live webcam feed with gesture overlays
            self.stream_camera()
            return
        if path == "/state":
            body = json.dumps({**ui_state, "messages": list(ui_state["messages"])}).encode()
            ctype = "application/json"
        elif path == "/planner":  # NEW: feeds the tasks panel (tasks, timers, alarms, stopwatches)
            body = json.dumps(snapshot()).encode()
            ctype = "application/json"
        elif path == "/news/categories":  # NEWS: category buttons in the News tab
            body = json.dumps(ada_news.categories_for_ui()).encode()
            ctype = "application/json"
        elif path == "/news":  # NEWS: headlines for the News tab (?cat=ai&q=openai&limit=40)
            body = json.dumps(ada_news.news_for_ui(parsed.query), ensure_ascii=False).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        else:
            try:
                with open(UI_FILE, "rb") as f:
                    body = f.read()
            except OSError:
                self.send_error(404, "ada_ui.html not found next to ada.py")
                return
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def stream_camera(self):
        if gesture is None:
            self.send_error(503, "gesture engine not running")
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        gesture.viewer_add(1)
        seq = -1
        try:
            while gesture.running:
                seq, jpg = gesture.wait_frame(seq)
                if jpg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        except OSError:
            pass  # browser closed the stream
        finally:
            gesture.viewer_add(-1)

    def log_message(self, *args):
        pass  # keep the console quiet

def start_ui_server():
    server = ThreadingHTTPServer(("127.0.0.1", UI_PORT), UIHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"UI running at http://localhost:{UI_PORT}")

_announced = {"listening": False}

def listen():
    """Records from the mic starting when speech is detected, and stops once
    the user has been silent for SILENCE_DURATION seconds - so it waits for
    you to actually finish talking instead of cutting off at a fixed length.
    Keeps a short rolling pre-buffer so the very start of your speech (e.g.
    the first syllable of a wake word) isn't clipped before onset is detected."""
    if not _announced["listening"]:
        print("Listening...")
        _announced["listening"] = True
    set_idle()
    PRE_BUFFER_SECONDS = 0.3
    pre_buffer_len = max(1, int(PRE_BUFFER_SECONDS * SAMPLE_RATE / BLOCK_SIZE))
    # extra room so the loud blocks that triggered onset are still in the buffer
    pre_buffer = deque(maxlen=pre_buffer_len + ONSET_BLOCKS)
    loud_run = 0

    chunks = []
    started = False
    silence_start = None
    record_start = None
    wait_start = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                         blocksize=BLOCK_SIZE) as stream:
        while True:
            block, _ = stream.read(BLOCK_SIZE)
            volume = np.abs(block).mean()
            now = time.time()

            if not started:
                if not gesture_cmds.empty():
                    return ""  # GESTURES: a gesture command is waiting - let the main loop handle it
                pre_buffer.append(block.copy())
                if volume > SILENCE_THRESHOLD:
                    loud_run += 1
                    if loud_run >= ONSET_BLOCKS:
                        # sustained sound = real speech, not a click or a bump
                        started = True
                        _announced["listening"] = False
                        set_state(mode="user")
                        record_start = now
                        chunks.extend(pre_buffer)  # lead-in + the blocks that triggered onset
                else:
                    loud_run = 0
                    if now - wait_start > PRE_SPEECH_TIMEOUT:
                        return ""  # nobody spoke - give up and let the caller loop again
                continue

            chunks.append(block.copy())

            if volume > SILENCE_THRESHOLD:
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = now
                elif now - silence_start > SILENCE_DURATION:
                    break  # been quiet long enough - done talking

            if now - record_start > MAX_RECORD_SECONDS:
                break  # safety cap

    if not chunks:
        return ""

    set_state(mode="thinking" if ui_state["awake"] else "asleep")
    audio = np.concatenate(chunks).flatten().astype(np.float32) / 32768.0

    # vad_filter drops silent/near-silent stretches before transcribing, which
    # is what stops Whisper from hallucinating words like "Okay." out of noise.
    # initial_prompt biases the decoder toward expecting "Ada" as a name, which
    # fixes it being misheard as "Aida"/"Eta"/"8 I"/etc.
    segments, _ = stt_model.transcribe(
        audio,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        initial_prompt="Ada is my voice assistant's name.",
    )
    text = " ".join(seg.text for seg in segments).strip()
    return text

# ---- TTS setup (Piper, British female voice) ----
print("Initializing TTS...")
PIPER_MODEL_PATH = "en_GB-jenny_dioco-medium.onnx"
piper_voice = PiperVoice.load(PIPER_MODEL_PATH)
print("TTS ready.")

INTERRUPT_THRESHOLD = 2500
INTERRUPT_GRACE_PERIOD = 0.6
interrupt_flag = threading.Event()
done_speaking = threading.Event()

def _interrupt_watcher_callback(indata, frames, time_, status):
    volume = np.abs(np.frombuffer(indata, dtype=np.int16)).mean()
    if volume > INTERRUPT_THRESHOLD:
        interrupt_flag.set()

def _watch_for_interrupt():
    time.sleep(INTERRUPT_GRACE_PERIOD)
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16',
                                channels=1, callback=_interrupt_watcher_callback):
            while not done_speaking.is_set():
                if interrupt_flag.is_set():
                    sd.stop()
                    break
                time.sleep(0.05)
    except Exception as e:
        print("Interrupt watcher error:", e)

last_spoken = {"text": ""}

def is_self_echo(heard):
    if not heard or not last_spoken["text"]:
        return False
    ratio = SequenceMatcher(None, heard.lower(), last_spoken["text"].lower()).ratio()
    return ratio > 0.6

def extract_spoken_answer(reply):
    match = re.search(r'ANSWER:\s*(.+)', reply, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # No ANSWER line (small models sometimes forget it): strip list markers and
    # headers so the reply reads naturally when spoken.
    text = re.sub(r'^\s*(?:[-*\u2022]|\d+[.)])\s+', '', reply, flags=re.MULTILINE)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    return re.sub(r'\n\s*\n+', '\n', text).strip()

def sanitize_for_speech(text):
    text = re.sub(r'\\\(|\\\)|\\\[|\\\]', '', text)
    text = text.replace('\\mod', 'mod')
    text = text.replace('\\', '')
    text = text.replace('**', '')
    text = text.replace('*', '')
    # NEWS: never read web addresses aloud
    text = re.sub(r'https?://\S+', '', text)
    return text

# NEW: alarms can go off while Ada is already talking, so only one thing speaks at a time.
speech_lock = threading.Lock()

def speak(text, shown=None):
    with speech_lock:
        return _speak(text, shown)

def _speak(text, shown=None):
    spoken_text = sanitize_for_speech(text)
    print("Speaking:", spoken_text)
    last_spoken["text"] = spoken_text
    set_state(mode="ada")
    add_message("ada", shown if shown is not None else spoken_text)
    interrupt_flag.clear()
    done_speaking.clear()

    watcher = threading.Thread(target=_watch_for_interrupt, daemon=True)
    watcher.start()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        piper_voice.synthesize_wav(spoken_text, wav_file)
    buffer.seek(0)

    with wave.open(buffer, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(n_frames)

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    sd.play(audio_np, samplerate=sample_rate)
    sd.wait()

    done_speaking.set()
    watcher.join(timeout=1)
    set_idle()

    was_interrupted = interrupt_flag.is_set()
    if was_interrupted:
        print("Interrupted by user speech.")
    return was_interrupted

# NEW: called by the background watcher when an alarm or timer goes off.
# It speaks even while Ada is asleep.
def announce_alert(alert):
    kind, label = alert["kind"], alert["label"]
    if label.strip().lower() == kind:
        text = f"Your {kind} is going off, sir."
    else:
        text = f"Your {label} {kind} is going off, sir."
    print("ALERT:", text)
    speak(text)

# NEW: drop the oldest messages, always restarting at a user message
def trim_history(history):
    while len(history) > MAX_HISTORY:
        history.pop(0)
        while history and history[0]["role"] != "user":
            history.pop(0)

# NEWS: planner tools go to schedualAGT, news tools go to ada_news
def run_any_tool(name, args):
    if ada_news.is_news_tool(name):
        return ada_news.run_news_tool(name, args, on_focus=news_focus)
    try:
        return run_tool(name, args)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

# NEW: now sends the tools and runs any tool calls Ada makes before returning her final reply
def ask_llm(prompt, history):
    start = len(history)
    # The model can't know the date, so it goes at the start of the user's message.
    # (Don't send a system message: it would override the Modelfile's SYSTEM prompt.)
    stamp = datetime.now().strftime("%A %Y-%m-%d %H:%M")
    history.append({"role": "user", "content": f"[{stamp}]\n{prompt}"})
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            resp = requests.post("http://localhost:11434/api/chat", json={
                "model": OLLAMA_MODEL,
                "messages": history,
                "tools": TOOLS,
                "stream": False,
                "options": {"num_ctx": OLLAMA_NUM_CTX}
            }, timeout=120)
            resp.raise_for_status()
            message = resp.json()["message"]
            history.append(message)

            calls = message.get("tool_calls")
            if not calls:
                trim_history(history)
                return message.get("content", "")

            for call in calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                result = run_any_tool(name, args)  # NEWS: was run_tool(name, args)
                print(f"Tool: {name}({args}) -> {result[:400]}")
                history.append({"role": "tool", "tool_name": name, "content": result})

        del history[start:]  # ran out of tool rounds - forget this turn
        return "Sorry, I got stuck using my tools. Could you try that again?"
    except requests.exceptions.RequestException as e:
        print("Error talking to Ollama:", e)
        del history[start:]  # don't leave a dangling user message behind
        return "Sorry, I couldn't reach the language model."

def normalize(text):
    # Lowercase and strip punctuation so "Alice, turn on." matches "alice turn on".
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def matches_command(text, phrases):
    text = normalize(text)
    if any(p in text for p in phrases):
        return True
    # Fuzzy fallback: catches near-miss transcriptions of short phrases
    # (e.g. "8 i turn on" instead of "ada turn on") that don't literally
    # contain the phrase but are close to it as a whole.
    words = text.split()
    for phrase in phrases:
        phrase_word_count = len(phrase.split())
        for i in range(max(1, len(words) - phrase_word_count + 1)):
            window = " ".join(words[i:i + phrase_word_count])
            if SequenceMatcher(None, window, phrase).ratio() > 0.75:
                return True
    return False

# ---- State machine: ASLEEP (only listens for wake phrase) / AWAKE ----
WAKE_PHRASES = ["ada turn on", "ada wake up"]
SLEEP_PHRASES = ["ada good night", "good night ada", "good night", "ada go to sleep"]
SHUTDOWN_PHRASES = ["ada shutdown system", "shutdown system", "ada shut down"]

def process_gesture_commands():
    """GESTURES: runs in the main thread. Handles commands queued by the gesture engine.
    Returns True if the whole system should shut down."""
    global awake
    while True:
        try:
            cmd = gesture_cmds.get_nowait()
        except queue.Empty:
            return False
        if cmd == "shutdown":
            speak("Shutting down. Goodbye, sir.")
            return True
        if cmd == "wake" and not awake:
            awake = True
            set_state(awake=True)
            interrupted = speak("I'm awake, sir.")
        elif cmd == "sleep" and awake:
            interrupted = speak("Good night, sir.")
            awake = False
            set_state(awake=False)
            set_idle()
        else:
            continue  # already in that state
        if not interrupted:
            time.sleep(SETTLE_TIME)

if __name__ == "__main__":
    print("Checking Ollama connection...")
    try:
        r = requests.get("http://localhost:11434", timeout=5)
        print("Ollama reachable:", r.status_code)
    except requests.exceptions.RequestException as e:
        print("WARNING: Ollama not reachable at localhost:11434 ->", e)
        print("Start it with 'ollama serve' in another terminal, or make sure the Ollama app is running.")

    # The system prompt is baked into the 'ada' model via its Modelfile,
    # so history starts empty (sending a system message here would override it).
    history = []

    start_ui_server()
    start_watcher(announce_alert)  # NEW: alarms and timers speak up on their own

    awake = False
    set_state(awake=False, mode="asleep")
    print("Ada is asleep. Say 'Ada turn on' (or double clap) to wake her.")

    # GESTURES: webcam + clap tracking. Callbacks only queue commands / move tabs;
    # the main loop below does the actual waking, sleeping and shutting down.
    gesture = GestureEngine(
        is_awake=lambda: awake,
        on_double_clap=lambda: gesture_cmds.put("wake"),   # first double clap only
        on_peace=lambda wake: gesture_cmds.put("wake" if wake else "sleep"),
        on_fist_close=lambda: gesture_cmds.put("shutdown"),
        on_swipe=tab_move,
    )
    gesture.start()

    while True:
        try:
            if process_gesture_commands():  # GESTURES
                break

            user_text = listen()
            if not user_text or len(user_text.strip()) < 2:
                continue  # silence/timeout or near-empty result - just listen again
            print("You said:", user_text)

            if is_self_echo(user_text):
                print("Ignoring likely self-echo:", user_text)
                continue

            # only show chatter in the UI when she's awake or it's a wake/shutdown command
            if awake or matches_command(user_text, WAKE_PHRASES + SHUTDOWN_PHRASES):
                add_message("user", user_text)

            if matches_command(user_text, SHUTDOWN_PHRASES):
                speak("Shutting down. Goodbye, sir.")
                break

            if not awake:
                if matches_command(user_text, WAKE_PHRASES):
                    awake = True
                    set_state(awake=True)
                    interrupted = speak("I'm awake, sir.")
                    if not interrupted:
                        time.sleep(SETTLE_TIME)
                continue

            if matches_command(user_text, SLEEP_PHRASES):
                interrupted = speak("Good night, sir.")
                awake = False
                set_state(awake=False)
                set_idle()
                if not interrupted:
                    time.sleep(SETTLE_TIME)
                continue

            set_state(mode="thinking")
            reply = ask_llm(user_text, history)
            print("Assistant:", reply)
            spoken_part = extract_spoken_answer(reply)
            shown = re.split(r'\s*ANSWER:', reply, flags=re.IGNORECASE)[0].strip() or spoken_part
            interrupted = speak(spoken_part, shown=shown)
            if not interrupted:
                time.sleep(SETTLE_TIME)
            # If interrupted, skip the settle pause and go straight back to
            # listening below - whatever the user is saying right now becomes
            # the next prompt.

        except Exception as e:
            import traceback
            print("MAIN LOOP ERROR:", e)
            traceback.print_exc()
            time.sleep(1)

    gesture.stop()  # GESTURES: release the webcam and mic before exiting
# =====================================================================
# INSTALLATION - DEPENDENCIES
# =====================================================================
#
# --- LINUX (Raspberry Pi OS / Debian / Ubuntu) ---
#
#   sudo apt update
#   sudo apt install -y python3-pip python3-venv portaudio19-dev libatlas-base-dev
#   pip install faster-whisper sounddevice numpy requests piper-tts openpyxl
#   pip install opencv-python mediapipe        # GESTURES
#
#   # Piper voice model (British female - Jenny):
#   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx
#   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json
#
#   # Ollama (LLM runtime) - installs a systemd service, run once:
#   curl -fsSL https://ollama.com/install.sh | sh
#   ollama create ada -f Modelfile
#
# --- WINDOWS ---
#
#   pip install faster-whisper sounddevice numpy requests piper-tts openpyxl
#   pip install opencv-python mediapipe        # GESTURES
#
#   (No separate PortAudio install needed - sounddevice ships a prebuilt
#   wheel with PortAudio bundled on Windows.)
#
#   # Piper voice model (British female - Jenny):
#   Download both files into the script folder:
#   https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx
#   https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json
#
#   # Ollama (LLM runtime):
#   Download and install from https://ollama.com/download
#   Then in a terminal:
#   ollama create ada -f Modelfile
#
# =====================================================================
