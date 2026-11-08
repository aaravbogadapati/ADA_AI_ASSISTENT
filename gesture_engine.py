"""
gesture_engine.py - hand-gesture / room tracker for Ada (ONE self-contained file).

The old hand_gesture_tracker_v19.py is built in below - you can delete that file.
Needs:  pip install opencv-python mediapipe sounddevice numpy

What it does
    * owns the webcam (the browser must NOT open it - the Camera tab shows our feed instead)
    * tracks you (only the person who did the 'both hands on chest' gesture is read)
    * calls back into Ada:
        on_double_clap()   -> called ONLY for the very first double clap after launch (wakes Ada).
                              Every later double clap just toggles gestures ON/OFF.
        on_peace(wake)     -> peace sign toggles Ada: asleep -> wake=True, awake -> wake=False.
                              It does NOT touch the gestures on/off state (only a double clap does)
        on_fist_close()    -> whole system shuts down (HOLD a closed fist for FIST_HOLD_SECONDS)
        on_swipe(+1 / -1)  -> UI moves to next / previous tab
                              RIGHT hand swipe (any direction) = previous tab (-1)
                              LEFT  hand swipe (any direction) = next tab     (+1)
        on_stop_speech()   -> hold an OPEN hand for OPEN_HOLD_SECONDS: Ada stops talking
    * NOTHING except the double clap is read until you double clap.
    * Hands are only read when they are ABOVE YOUR NOSE (centre of your face).
    * keeps the latest annotated frame as a JPEG so ada.py can stream it to the Camera tab
      (frames are only drawn and encoded while the Camera tab is open)
"""

import math
import os
import threading
import time
import traceback
import urllib.request
from collections import Counter, deque, namedtuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---- Ada / engine settings -------------------------------------------------------------
CAMERA_INDEX = 0
GESTURES_START_ON = False        # False = peace / fist / swipe are ignored until your first double clap
PEACE_COOLDOWN = 1.5             # seconds between peace-sign triggers
LOCKOUT_AFTER_CLAP = 1.0         # ignore other gestures this long after a double clap
FIST_HOLD_SECONDS = 5.0          # hold a closed fist this long to shut the system down
OPEN_HOLD_SECONDS = 3.0          # hold an OPEN hand this long to interrupt Ada and stop her speaking
OPEN_GRACE = 0.4                 # the open hand may flicker for this long without restarting the countdown
FIST_GRACE = 0.4                 # the fist may flicker for this long without restarting the countdown
SWIPE_ANY_MOVE = True            # open palm swiped in ANY direction (up/down/sideways) also counts, not just turning
SWIPE_MOVE_MIN = 3.0             # how far the palm must travel, in hand-lengths, within ~0.5s
SWIPE_MOVE_COOLDOWN = 1.0        # seconds between movement swipes (also ignores the hand coming back)
SWAP_HANDS = False               # set True if the labels come out reversed (your left hand reads as Right)
STREAM_ZOOM_PANEL = False        # True = the UI feed also shows the zoom panel
JPEG_QUALITY = 70
SHOW_WINDOW = False              # True = also open the normal OpenCV window
# ---------------------------------------------------------------------------


# =====================================================================
#  TRACKING CORE (formerly hand_gesture_tracker_v19.py)
# =====================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
POSE_MODEL_NAME = "pose_landmarker_lite.task"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

WRIST, MIDDLE_MCP = 0, 9
FINGERS = [(8, 5), (12, 9), (16, 13), (20, 17)]   # (tip, mcp): index, middle, ring, pinky

# ---- Tunable thresholds -------------------------------------------------
# Finger detection (tip-to-wrist distance / knuckle-to-wrist distance)
EXT_ON = 1.40                 # above this -> finger is EXTENDED
EXT_OFF = 1.25                # below this -> finger is CURLED (between = keep previous)

# Gesture stability
DEBOUNCE = 0.25               # seconds a new gesture must be held before it switches
TRACK_GRACE = 0.7             # seconds a hand may vanish before it is forgotten
TRACK_MATCH_DIST = 0.30       # max wrist jump (fraction of frame) to count as same hand

# Rotation (angles in degrees, positions as fraction of frame size)
ROT_WINDOW = 1.0
ROT_HOLD_DEG = 25             # turned this much while barely moving -> HOLD gesture
ROT_HOLD_MOVE = 0.08
FIST_ROT_DEG = 90             # fist must turn this much to fire "Fist Rotate"
FIST_ROT_MOVE = 0.12
ROT_COOLDOWN = 1.0
FIST_CLOSE_COOLDOWN = 0.8

# Clap (gap = distance between palm centres / palm length)
CLAP_CONTACT = 1.10           # a dip in the gap below this = hands touched (bigger = more forgiving)
CLAP_BOUNCE = 0.15            # the gap must bounce back out by this much to confirm the touch
CLAP_LOST_GAP = 3.0           # hands vanish (overlap/blur) while closing from within this gap
DOUBLE_CLAP_WINDOW = 0.5      # max seconds between touch 1 and touch 2
DOUBLE_CLAP_COOLDOWN = 0.3    # seconds after a double clap before another can fire

# Sound double-check (pip install sounddevice numpy)
USE_AUDIO = True              # False = double clap is judged by the camera only
AUDIO_RATE = 16000            # microphone sample rate
AUDIO_MIN_LEVEL = 0.02        # a clap must be at least this loud
AUDIO_SPIKE_RATIO = 5.0       # ...and this many times louder than the room's background noise
AUDIO_ONSET_RATIO = 2.5       # ...and suddenly louder than the moment before (so steady noise is ignored)
AUDIO_REFRACTORY = 0.06       # one clap = one spike; ignore spikes this soon after another
AUDIO_DOUBLE_WINDOW = 0.6     # two clap sounds must be within this many seconds
AUDIO_SYNC = 0.4              # a clap sound must be within this many seconds of a hand touch
AUDIO_FILLS_IN_ONE_TOUCH = True   # camera saw only ONE touch but mic heard two claps -> still counts
FUSE_WINDOW = 1.0             # how far back (seconds) touches and sounds are considered together

# Multiple people
DUP_IOU = 0.55                # two detections whose body boxes overlap this much (IoU) = the same person
DUP_INSIDE = 0.85             # ...or one box is this much inside the other AND their centres are close
DUP_CENTER = 0.15             #   (centre distance, fraction of frame)
REID_IOU = 0.30               # a new detection overlapping a person we just lost by this much = same person
HAND_MAX = 4                  # hands the hand detector looks for (more people = more hands)
AUTO_ID_ONLY_WHEN_ALONE = True    # if 2+ people are in view, only the chest gesture can pick who you are
LOCK_TO_CLAIMED = True        # after the chest gesture, only the chest gesture (or the same person) can pick 'you'

# Forgiveness
SHAPE_VOTE_WINDOW = 0.25      # seconds of recent frames that vote on the hand shape
FIST_CLOSE_LOOKBACK = 1.2     # Fist Close fires if the hand was open within this many seconds
GESTURE_MARGIN_STICKY = 0.05  # a hand already being read may sink this far below the nose line
STICKY_TIME = 0.4             # ...if it was read within this many seconds
SHOULDER_SMOOTH = 0.3         # 0..1 per update, lower = steadier shoulder line

# Swipe (open palm)
SWIPE_BY_ROTATION = True      # turn your open hand left/right
SWIPE_ROT_DEG = 30            # how far the hand must turn (degrees) within ROT_WINDOW seconds
SWIPE_OPEN_GRACE = 0.2        # seconds the fingers may flicker out of 'open' mid-turn without cancelling
SWIPE_WINDOW = 0.5            # seconds a movement swipe must happen within
SWIPE_MIN_SAMPLES = 4         # frames needed inside the window
SWIPE_COOLDOWN = 0.5          # seconds between swipes
SWIPE_RETURN_GUARD = 1.0      # seconds the OPPOSITE turn direction is ignored (your hand coming back)

# Shoulder gating / hand ownership
GESTURE_MARGIN = 0.02         # palm may sit this far BELOW the nose line and still count (frame fraction)
GESTURE_IF_NO_SHOULDERS = False   # if your shoulders can't be seen: True = read all hands, False = read none
SHOULDER_MIN_VIS = 0.3        # shoulder-point confidence needed to compute the shoulder line
SHOULDER_MAX_AGE = 2.0        # seconds the last known shoulder line stays valid
HAND_OWN_DIST = 0.20          # a hand this close to your pose wrist/hand points is yours
HAND_REACH = 3.2              # otherwise: within this many shoulder-widths of your shoulder centre
HAND_LINK_DIST = 0.15         # pose wrist this close to a detected hand = that body "has" that hand
HAND_LINK_CONFIRM = 0.5       # seconds a hand must stay attached before the body is accepted as you
HAND_LINK_GRACE = 0.4         # brief dropouts allowed while linked
HAND_LINK_ACCEPTS = True      # a body with a visible hand attached is accepted as you when you're away/hidden

# Chest claim (both hands on chest = this body is me)
CHEST_HOLD = 0.6              # seconds both hands must stay on your chest
CHEST_GRACE = 0.6             # dropout allowed during the hold
CHEST_COOLDOWN = 3.0          # seconds before another claim can fire
CHEST_X_PAD = 0.40            # chest box extends this many shoulder-widths past each shoulder
CHEST_DEPTH = 0.90            # chest box goes this fraction of the way from shoulders down to hips

# Camera / zoom
CAMERA_W, CAMERA_H = 1280, 720   # ask the webcam for this size (lower it, e.g. 640x480, to save CPU / RAM)
FPS_BOOST = 1.2               # ask the camera for this multiple of its default frame rate
PANEL_W = 640                 # width of the picture streamed to the UI
ZOOM_ENABLED = True           # False = hands are detected on the full view only
ZOOM_SHOULDER_FRAC = 0.22     # zoom in until your shoulders are this fraction of the crop width
ZOOM_MAX = 3.0                # never zoom in more than this
ZOOM_ENGAGE = 1.15            # below this zoom, just stay on the full view
ZOOM_UP = 0.10                # shift the crop up (fraction of crop height) so raised hands fit
ZOOM_SMOOTH = 0.08            # 0..1 per frame, lower = smoother / slower zoom
ZOOM_HOLD = 1.5               # seconds to keep the zoom after losing you, then zoom back out

# Room presence
POSE_EVERY_N = 3              # run the body tracker every Nth frame (saves CPU)
POSE_MIN_VIS = 0.5            # body-point confidence needed to FIND you (need 2 of nose/shoulders)
POSE_KEEP_VIS = 0.25          # lower bar once locked on, so partial blocking keeps tracking
POS_SMOOTHING = 0.5           # 0..1, higher = follows the raw position more tightly
ENTER_CONFIRM = 0.8           # seconds you must be seen steadily to count as 'entered'
DISAPPEAR_GRACE = 2.0         # seconds with no sign of you before we decide Left vs Hidden
EDGE_HARD = 0.06              # vanished this close to an edge = you walked out
MIN_EXIT_SPEED = 0.08         # frame-widths/sec toward an edge to count as heading out
EXIT_LOOKAHEAD = 1.2          # heading out if you'd reach that edge within this many seconds
DOOR_ZONES = []               # optional doorways in the middle of the view, e.g. [(0.4, 0.1, 0.6, 0.9)]
HIDDEN_MAX = 30 * 60          # seconds hidden before giving up and saying Left (0 = never)
RESUME_CONFIRM = 0.4          # seconds your body must be seen again to resume from hidden
EDGE_MARGIN = 0.30            # something must first appear within this fraction of an edge to count as entering
RETURN_SAME_SIDE_ONLY = True  # after you leave via an edge, only accept a return from that SAME edge
POSE_MAX_PEOPLE = 2           # body-shaped things tracked at once - set to the number of people in the frame
CAND_GATE_BASE = 0.12         # how far a thing can move between detections and still be the same thing
CAND_GATE_GROW = 0.30         #   (+ this much per second since it was last seen)
CAND_GATE_MAX = 0.40
CAND_EXPIRE = 1.5             # seconds unseen before an identity track is dropped
DECOY_SECONDS = 4.0           # seen this long without moving = a static lump, never counted as you
DECOY_MAX_DISP = 0.05         # 'without moving' = stayed within this fraction of the frame
ACQUIRE_MIN_MOVE = 0.06       # something must really MOVE this much to count as entering / reappearing
HIDDEN_RESUME_RADIUS = 0.45   # a reappearance must be this close to where you hid
HAND_NEAR = 0.35              # a still body counts as you if one of your hands is this close to it
ENTRY_BACKTRACK = 1.0         # if first spotted late, project its path back this many seconds to find its edge
FALLBACK_SECONDS = 3.0        # (only if you never left via an edge) something in view this long...
FALLBACK_MOVE = 0.25          # ...that has travelled this far across the frame is accepted as you
STARTUP_SILENT = 3.0          # if you're already in view at launch, don't fire 'Entered'
ROOM_EVENT_DISPLAY = 3.0      # seconds a room event stays on screen
TRAIL_LEN = 40                # length of the position trail

EVENT_DISPLAY = 1.2           # seconds an event stays on screen
# --------------------------------------------------------------------------


def ensure_model(name, url):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, path)
        print("Done.")
    return path


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_ratios(lm):
    """Per finger: tip-to-wrist / knuckle-to-wrist. ~0.7-1.2 curled, ~1.6-2.2 extended."""
    wrist = lm[WRIST]
    return [dist(lm[t], wrist) / (dist(lm[m], wrist) + 1e-6) for t, m in FINGERS]


def hand_size(lm):
    return dist(lm[WRIST], lm[MIDDLE_MCP]) + 1e-6


def palm_center(lm):
    idx = (0, 5, 9, 13, 17)
    return (sum(lm[i].x for i in idx) / 5, sum(lm[i].y for i in idx) / 5)


def hand_angle(lm):
    return math.atan2(lm[MIDDLE_MCP].y - lm[WRIST].y, lm[MIDDLE_MCP].x - lm[WRIST].x)


def shape_from_fingers(ext):
    """ext = [index, middle, ring, pinky] booleans -> raw shape ('Other' = unclear)."""
    index, middle, ring, pinky = ext
    up = sum(ext)
    if up == 0:
        return "Fist"
    if index and not (middle or ring or pinky):
        return "Pointing"
    if index and middle and not (ring or pinky):
        return "Peace"
    if up >= 3:
        return "Open Hand"
    return "Other"


class HandTrack:
    """One tracked hand: hysteresis fingers, debounced gesture, rotation tracking."""

    _next_id = 1

    def __init__(self, name, wrist_pos, now):
        self.id = HandTrack._next_id
        HandTrack._next_id += 1
        self.name = name
        self.last_pos = wrist_pos
        self.last_seen = now

        self.ext = [False] * 4
        self.stable = None            # the one live gesture
        self.cand = None              # candidate gesture waiting out the debounce
        self.cand_since = 0.0

        self.history = deque()        # (t, unwrapped_angle, x, y)
        self.prev_angle = None
        self.unwrapped = 0.0
        self.rotating_in_place = False
        self.last_rot_event = -1e9
        self.last_close_event = -1e9
        self.shape_hist = deque()     # (t, raw_shape) votes
        self.shape = "Other"          # voted shape
        self.last_open_time = -1e9    # last time the voted shape was an open hand

        self.swipe_hist = deque()     # (t, palm_x, palm_y, hand_size) while the palm is open
        self.last_swipe = -1e9
        self.last_swipe_dir = 0
        self.open_since = None        # when the palm last became (and stayed) open
        self.last_open = -1e9

    # -- swipe by turning the hand ---------------------------------------------
    def _check_swipe_rotation(self, now, turned):
        if self.shape == "Open Hand":
            self.last_open = now
            if self.open_since is None:
                self.open_since = now
        elif self.open_since is None or now - self.last_open > SWIPE_OPEN_GRACE:
            self.open_since = None                        # really not an open palm any more
            return None
        if not self.history or self.open_since > self.history[0][0]:
            return None                                   # palm wasn't open for the whole turn
        if abs(turned) < SWIPE_ROT_DEG:
            return None
        direction = 1 if turned > 0 else -1               # + = clockwise on screen
        since = now - self.last_swipe
        if since < SWIPE_COOLDOWN:
            return None
        if direction == -self.last_swipe_dir and since < SWIPE_RETURN_GUARD:
            return None                                   # that's just turning back
        self.last_swipe, self.last_swipe_dir = now, direction
        self.history.clear()
        self.unwrapped = 0.0
        return "Swipe Right" if direction > 0 else "Swipe Left"

    # -- rotation bookkeeping ---------------------------------------------
    def _track_rotation(self, now, lm):
        a = hand_angle(lm)
        if self.prev_angle is not None:
            d = (a - self.prev_angle + math.pi) % (2 * math.pi) - math.pi
            self.unwrapped += d
        self.prev_angle = a
        self.history.append((now, self.unwrapped, lm[WRIST].x, lm[WRIST].y))
        while self.history and now - self.history[0][0] > ROT_WINDOW:
            self.history.popleft()
        if len(self.history) < 4:
            return 0.0, 1.0
        # biggest swing in either direction anywhere in the window (not just first -> last)
        up = down = 0.0
        lo = hi = self.history[0][1]
        for _, ang, _, _ in self.history:
            lo, hi = min(lo, ang), max(hi, ang)
            up = max(up, ang - lo)       # biggest clockwise swing ending here
            down = min(down, ang - hi)   # biggest counter-clockwise swing ending here
        turned = math.degrees(up if up >= -down else down)
        x0, y0 = self.history[0][2], self.history[0][3]
        travel = max(math.hypot(x - x0, y - y0) for _, _, x, y in self.history)
        return turned, travel

    # -- finger hysteresis ------------------------------------------------
    def _update_fingers(self, lm):
        for i, r in enumerate(finger_ratios(lm)):
            if r > EXT_ON:
                self.ext[i] = True
            elif r < EXT_OFF:
                self.ext[i] = False
            # in between: keep previous state (hysteresis)

    # -- debounced gesture switch ----------------------------------------
    def _debounce(self, now, raw):
        """Returns True only when the live gesture actually changes."""
        if raw == "Other" or raw == self.stable:
            self.cand = None          # unclear / same as live gesture -> nothing pending
            return False
        if raw != self.cand:
            self.cand, self.cand_since = raw, now
            return False
        if now - self.cand_since >= DEBOUNCE:
            self.stable, self.cand = raw, None
            return True
        return False

    # -- main update --------------------------------------------------------
    def update(self, now, lm):
        """Returns (live_gesture, changed, event_or_None)."""
        self.last_pos = (lm[WRIST].x, lm[WRIST].y)
        self.last_seen = now

        turned, travel = self._track_rotation(now, lm)
        self.rotating_in_place = abs(turned) >= ROT_HOLD_DEG and travel < ROT_HOLD_MOVE

        self._update_fingers(lm)
        self.shape_hist.append((now, shape_from_fingers(self.ext)))
        while self.shape_hist and now - self.shape_hist[0][0] > SHAPE_VOTE_WINDOW:
            self.shape_hist.popleft()
        votes = Counter(r for _, r in self.shape_hist if r != "Other")
        self.shape = votes.most_common(1)[0][0] if votes else "Other"
        if self.shape == "Open Hand":
            self.last_open_time = now
        changed = False
        if not (self.rotating_in_place and self.stable is not None):
            changed = self._debounce(now, self.shape)

        event = None
        # Fist Close: an OPEN hand became a fist (registered once)
        if (changed and self.stable == "Fist"
                and now - self.last_open_time < FIST_CLOSE_LOOKBACK
                and now - self.last_close_event > FIST_CLOSE_COOLDOWN):
            event = "Fist Close"
            self.last_close_event = now

        # Fist Rotate: fist turning in place
        if (event is None and self.stable == "Fist"
                and abs(turned) >= FIST_ROT_DEG and travel < FIST_ROT_MOVE
                and now - self.last_rot_event > ROT_COOLDOWN):
            event = "Fist Rotate " + ("Clockwise" if turned > 0 else "Counter-Clockwise")
            self.last_rot_event = now
            self.history.clear()
            self.unwrapped = 0.0

        if event is None and SWIPE_BY_ROTATION:
            event = self._check_swipe_rotation(now, turned)

        return self.stable, changed, event


class ClapDetector:
    """Finds the moments your two hands come together (a 'touch').
    A touch = the gap between your palms dips below CLAP_CONTACT and then bounces back
    out by CLAP_BOUNCE. DoubleClapDecider decides whether two touches make a DOUBLE clap
    (it also listens to the microphone)."""

    def __init__(self):
        self.touches = deque()        # times of recent touches
        self.hist = deque()           # (t, gap) while two hands visible
        self.mode = None              # zig-zag tracker: "up" / "down"
        self.ext = 0.0                # current extreme of the gap (peak when up, valley when down)
        self.ext_t = 0.0
        self.peak = 0.0               # last peak before the current valley
        self.last_two_time = 0.0
        self.prev_count = 0
        self.gap = None

    def _touch(self, t):
        self.touches.append(t)
        while self.touches and t - self.touches[0] > 3.0:
            self.touches.popleft()

    def _was_closing(self):
        if not self.hist:
            return False
        last = self.hist[-1][1]
        if last > CLAP_LOST_GAP:
            return False
        if last < .5:
            return True
        return self.hist[0][1] > last * 1.15      # gap was shrinking

    def update(self, now, hands):
        self.gap = None
        if len(hands) == 2:
            c1, c2 = palm_center(hands[0]), palm_center(hands[1])
            size = (hand_size(hands[0]) + hand_size(hands[1])) / 2
            gap = math.hypot(c1[0] - c2[0], c1[1] - c2[1]) / size
            self.gap = gap

            self.hist.append((now, gap))
            while self.hist and now - self.hist[0][0] > 0.4:
                self.hist.popleft()

            if self.mode is None:
                self.mode, self.ext, self.ext_t, self.peak = "up", gap, now, gap
            elif self.mode == "down":                 # following the gap down to a valley
                if gap < self.ext:
                    self.ext, self.ext_t = gap, now
                elif gap - self.ext >= CLAP_BOUNCE:   # bounced back out -> that valley was a touch
                    valley, vt = self.ext, self.ext_t
                    self.mode, self.ext, self.ext_t = "up", gap, now
                    if valley < CLAP_CONTACT and self.peak - valley >= CLAP_BOUNCE:
                        self._touch(vt)
            else:                                     # following the gap up to a peak
                if gap > self.ext:
                    self.ext, self.ext_t = gap, now
                elif self.ext - gap >= CLAP_BOUNCE:
                    self.peak = self.ext
                    self.mode, self.ext, self.ext_t = "down", gap, now
            self.last_two_time = now
        else:
            # Palms overlapping often makes the tracker drop or merge the hands
            if self.prev_count == 2 and self.mode == "down" and self._was_closing():
                self._touch(now)
                self.mode = None
            if now - self.last_two_time > 0.4:
                self.hist.clear()
                self.mode = None

        self.prev_count = len(hands)


class AudioClapListener:
    """Listens on the microphone for sharp clap-like spikes (runs in its own thread)."""

    def __init__(self):
        self.ok = False
        self.error = None
        self.spikes = deque(maxlen=30)    # times of clap-like sounds
        self.level = 0.0                  # latest loudness
        self._bg = 0.003                  # slow estimate of background noise
        self._prev = 0.0
        self._last = -1e9
        self._np = None
        self._stream = None

    def start(self):
        if not USE_AUDIO:
            self.error = "disabled (USE_AUDIO = False)"
            return
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            self.error = "not installed - run: pip install sounddevice numpy"
            return
        try:
            self._np = np
            self._stream = sd.InputStream(samplerate=AUDIO_RATE, channels=1,
                                          blocksize=int(AUDIO_RATE * 0.01),   # ~10 ms blocks
                                          callback=self._callback)
            self._stream.start()
            self.ok = True
        except Exception as e:                                        # no mic / busy / blocked
            self.error = str(e)

    def _callback(self, indata, frames, time_info, status):
        np = self._np
        x = indata[:, 0]
        lvl = float(np.sqrt(np.mean(np.diff(x) ** 2)))                # high-pass: claps are sharp
        now = time.time()
        loud = lvl > max(AUDIO_MIN_LEVEL, AUDIO_SPIKE_RATIO * self._bg)
        onset = lvl > AUDIO_ONSET_RATIO * max(self._prev, 1e-6)
        if loud and onset and now - self._last > AUDIO_REFRACTORY:
            self._last = now
            self.spikes.append(now)
        self._bg += (0.002 if loud else 0.02) * (lvl - self._bg)      # background adapts slowly when loud
        self._prev = lvl
        self.level = lvl

    def recent(self, now, window):
        return [t for t in list(self.spikes) if now - t <= window]

    def clear(self):
        self.spikes.clear()

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass


class DoubleClapDecider:
    """Combines the camera (hand touches) and the microphone (clap sounds)."""

    def __init__(self):
        self.last_double = -1e9

    def check(self, now, clap, audio):
        if now - self.last_double < DOUBLE_CLAP_COOLDOWN:
            return False
        touches = sorted(t for t in clap.touches if now - t <= FUSE_WINDOW)
        video_double = any(b - a <= DOUBLE_CLAP_WINDOW for a, b in zip(touches, touches[1:]))
        if audio is not None and audio.ok:
            sounds = sorted(audio.recent(now, FUSE_WINDOW))
            if not any(b - a <= AUDIO_DOUBLE_WINDOW for a, b in zip(sounds, sounds[1:])):
                return False                                          # didn't hear two claps
            synced = any(abs(sd_ - t) <= AUDIO_SYNC for sd_ in sounds for t in touches)
            fire = synced and (video_double or (AUDIO_FILLS_IN_ONE_TOUCH and len(touches) >= 1))
        else:
            fire = video_double                                       # no mic: camera only
        if fire:
            self.last_double = now
            clap.touches.clear()
            if audio is not None:
                audio.clear()
        return fire


POSE_NOSE, POSE_LSH, POSE_RSH, POSE_LHIP, POSE_RHIP = 0, 11, 12, 23, 24
POSE_WRISTS = (15, 16)
POSE_HAND_POINTS = (15, 16, 17, 18, 19, 20, 21, 22)   # wrists, pinkies, indexes, thumbs
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (0, 11), (0, 12),
]


def body_center(pose_lms, tracking=False):
    """(x, y) centre of your head/torso, or None if your body isn't visible enough.
    Once already tracking, the confidence bar is lower so partial blocking is tolerated."""
    min_vis = POSE_KEEP_VIS if tracking else POSE_MIN_VIS
    need = 1 if tracking else 2

    def vis(i):
        v = pose_lms[i].visibility
        return v if v is not None else 0.0

    if sum(1 for i in (POSE_NOSE, POSE_LSH, POSE_RSH) if vis(i) > min_vis) < need:
        return None
    pts = [pose_lms[i] for i in (POSE_NOSE, POSE_LSH, POSE_RSH)      # no hips: they flicker in and out
           if vis(i) > min_vis]
    return (sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts))


def skeleton_from(pose_lms):
    return [(p.x, p.y, p.visibility if p.visibility is not None else 0.0) for p in pose_lms]


def draw_skeleton(frame, skel, color, min_vis=0.3):
    h, w, _ = frame.shape
    for a, b in POSE_CONNECTIONS:
        pa, pb = skel[a], skel[b]
        if pa[2] > min_vis and pb[2] > min_vis:
            cv2.line(frame, (int(pa[0] * w), int(pa[1] * h)), (int(pb[0] * w), int(pb[1] * h)),
                     color, 2, cv2.LINE_AA)
    for x, y, v in skel:
        if v > min_vis:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, color, -1, cv2.LINE_AA)


def skel_box(skel, min_vis=0.3):
    """Bounding box (x0, y0, x1, y1) of a skeleton's visible points, or None."""
    pts = [(x, y) for x, y, v in skel if v > min_vis]
    if len(pts) < 3:
        return None
    xs, ys = zip(*pts)
    return (min(xs), min(ys), max(xs), max(ys))


def box_overlap(a, b):
    """(IoU, share of the smaller box that lies inside the other) for two boxes."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0, 0.0
    inter = iw * ih
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
    return inter / (area_a + area_b - inter), inter / min(area_a, area_b)


def dedupe_detections(detections):
    """Merge detections that are really the same body (pose model sometimes returns two).
    Keeps the best-seen one of each overlapping group."""
    keep = []
    for pos, skel in sorted(detections, key=lambda d: -sum(p[2] for p in d[1])):
        box = skel_box(skel)
        dup = False
        for kpos, kskel, kbox in keep:
            dc = math.hypot(pos[0] - kpos[0], pos[1] - kpos[1])
            if box is not None and kbox is not None:
                iou, inside = box_overlap(box, kbox)
                dup = iou > DUP_IOU or (inside > DUP_INSIDE and dc < DUP_CENTER)
            else:
                dup = dc < DUP_CENTER / 2
            if dup:
                break
        if not dup:
            keep.append((pos, skel, box))
    return [(pos, skel) for pos, skel, _ in keep]


def shoulder_center(skel, min_vis=0.2):
    """(cx, cy, width) of the shoulders in a skeleton, or None if both aren't visible."""
    l, r = skel[POSE_LSH], skel[POSE_RSH]
    if l[2] < min_vis or r[2] < min_vis:
        return None
    return ((l[0] + r[0]) / 2, (l[1] + r[1]) / 2,
            max(math.hypot(l[0] - r[0], l[1] - r[1]), 0.02))


def chest_box(skel):
    """(x0, y0, x1, y1) of the chest area for this skeleton, or None."""
    sc = shoulder_center(skel)
    if sc is None:
        return None
    cx, cy, width = sc
    lh, rh = skel[POSE_LHIP], skel[POSE_RHIP]
    if lh[2] > 0.2 and rh[2] > 0.2:
        hip_y = (lh[1] + rh[1]) / 2
    else:
        hip_y = cy + 1.6 * width                  # estimate when hips are out of view
    half = width * (0.5 + CHEST_X_PAD)
    return (cx - half, cy - 0.15 * width, cx + half, cy + CHEST_DEPTH * (hip_y - cy))


def skeleton_has_hand_near(skel, hand_points):
    """True if one of this body's pose wrists sits on top of a detected hand."""
    for hx, hy in hand_points:
        for i in POSE_WRISTS:
            x, y, v = skel[i]
            if v > 0.2 and math.hypot(x - hx, y - hy) < HAND_LINK_DIST:
                return True
    return False


P = namedtuple("P", "x y")    # a landmark point mapped back to full-frame coordinates


class ZoomView:
    """Smoothly follows your upper body with a digital-zoom crop (aspect ratio preserved).
    The rect is (x0, y0, w, h) as fractions of the full frame."""

    def __init__(self):
        self.cx = self.cy = 0.5
        self.z = 1.0
        self.last_target = -1e9

    def update(self, now, body):
        """body = (shoulder_cx, shoulder_cy, shoulder_width) of YOU, or None."""
        if not ZOOM_ENABLED:
            return
        if body is not None:
            cx, sy, sw = body
            tz = min(max(ZOOM_SHOULDER_FRAC / sw, 1.0), ZOOM_MAX)
            if tz < ZOOM_ENGAGE:
                tz = 1.0
            tcx, tcy = cx, sy - ZOOM_UP / tz
            self.last_target = now
        elif now - self.last_target < ZOOM_HOLD:
            return
        else:
            tz, tcx, tcy = 1.0, 0.5, 0.5
        a = ZOOM_SMOOTH
        self.z += a * (tz - self.z)
        self.cx += a * (tcx - self.cx)
        self.cy += a * (tcy - self.cy)

    def rect(self):
        size = 1.0 / self.z
        x0 = min(max(self.cx - size / 2, 0.0), 1.0 - size)
        y0 = min(max(self.cy - size / 2, 0.0), 1.0 - size)
        return x0, y0, size, size


def nearest_edge(pos):
    """'left' / 'right' / 'top' / 'bottom' if pos is close to that image edge, else None."""
    x, y = pos
    d = {"left": x, "right": 1 - x, "top": y, "bottom": 1 - y}
    side = min(d, key=d.get)
    return side if d[side] < EDGE_MARGIN else None


class CandTrack:
    """One body-shaped thing the pose model keeps seeing: you, or a lump of clothes..."""

    _next_id = 1

    def __init__(self, pos, skel, now):
        self.id = CandTrack._next_id
        CandTrack._next_id += 1
        self.pos, self.skel = pos, skel
        self.first_pos, self.first_seen = pos, now
        self.run_pos, self.run_start = pos, now      # start of the current continuous sighting
        self.last_seen = now
        self.max_disp = 0.0
        self.link_since = None                       # when a visible hand became attached to this body
        self.link_last = -1e9

    def update(self, pos, skel, now):
        if now - self.last_seen > 0.5:               # seen again after a gap -> new sighting run
            self.run_pos, self.run_start = pos, now
        a = POS_SMOOTHING
        self.pos = (a * pos[0] + (1 - a) * self.pos[0], a * pos[1] + (1 - a) * self.pos[1])
        self.skel = skel
        self.last_seen = now
        self.max_disp = max(self.max_disp, math.hypot(self.pos[0] - self.first_pos[0],
                                                      self.pos[1] - self.first_pos[1]))

    def run_move(self):
        return math.hypot(self.pos[0] - self.run_pos[0], self.pos[1] - self.run_pos[1])

    def is_static(self, now):
        """Has been in view a while without really moving = clutter, not a person."""
        return now - self.first_seen >= DECOY_SECONDS and self.max_disp < DECOY_MAX_DISP

    # -- hand attachment ---------------------------------------------------
    def update_link(self, now, hand_points):
        if skeleton_has_hand_near(self.skel, hand_points):
            if now - self.link_last > HAND_LINK_GRACE:
                self.link_since = now
            self.link_last = now

    def hand_linked(self, now):
        """A visible hand has been attached to this body for a moment = it's a person."""
        return (self.link_since is not None and now - self.link_last < HAND_LINK_GRACE
                and now - self.link_since >= HAND_LINK_CONFIRM)


class PresenceTracker:
    """ABSENT -> PRESENT -> (HIDDEN | ABSENT), following ONE target identity.

    * Only the identity track that has been following you counts as you.
    * A body with a visible hand attached to it is accepted as you.
    * Both hands on your chest (held briefly) = "this body is me".
    * Vanishing near an edge while heading out = you left (ABSENT).
    * Vanishing mid-view = HIDDEN (still in the room) until something MOVES into
      view near where you hid."""

    ABSENT, PRESENT, HIDDEN = "ABSENT", "PRESENT", "HIDDEN"

    def __init__(self, start_time):
        self.start = start_time
        self.state = self.ABSENT
        self.state_since = start_time
        self.last_evidence = -1e9     # last sign of you (your track or hands)
        self.last_body = -1e9         # last time YOUR track was seen
        self.hands_since = None
        self.expect_edge = None       # edge we're waiting for you to come back through
        self.target_id = None
        self.cands = []
        self.last_pos = None
        self.skeleton = None
        self.motion = deque(maxlen=12)    # (t, x, y) of your track
        self.trail = deque(maxlen=TRAIL_LEN)
        self.chest_id = None          # candidate currently holding the chest pose
        self.chest_since = 0.0
        self.chest_last = -1e9
        self.last_claim = -1e9
        self.sh_ema = None            # smoothed shoulder line
        self.sh_t = -1e9
        self.nose_ema = None          # smoothed nose line
        self.nose_t = -1e9
        self.people_count = 0         # people currently in view
        self.auto_ok = True           # may we pick 'you' automatically right now?
        self.locked = False           # someone has claimed control with the chest gesture

    @property
    def present(self):
        return self.state != self.ABSENT      # hidden still counts as in the room

    # -- public helpers -------------------------------------------------------
    def near_target(self, pos):
        """True if pos is close to where you were (used to accept lower-confidence body points)."""
        return (pos is not None and self.state != self.ABSENT and self.last_pos is not None
                and math.hypot(pos[0] - self.last_pos[0], pos[1] - self.last_pos[1]) < 0.3)

    def ignored(self, now):
        """Body-shaped things currently in view that are NOT you."""
        return [t for t in self.cands if t.id != self.target_id and now - t.last_seen < 0.3]

    def shoulder_line(self, now):
        """y (fraction of frame) of YOUR shoulders, or None if unknown / stale."""
        if (self.state != self.PRESENT or self.skeleton is None
                or now - self.last_body > SHOULDER_MAX_AGE):
            return None
        ys = [self.skeleton[i][1] for i in (POSE_LSH, POSE_RSH)
              if self.skeleton[i][2] > SHOULDER_MIN_VIS]
        if not ys:
            return None
        raw = sum(ys) / len(ys)
        if self.sh_ema is None or now - self.sh_t > 1.0:
            self.sh_ema = raw
        else:
            self.sh_ema += SHOULDER_SMOOTH * (raw - self.sh_ema)
        self.sh_t = now
        return self.sh_ema

    def nose_line(self, now):
        """y (fraction of frame) of YOUR nose (centre of face), or None if unknown / stale.
        Gestures are only read when the hand is above this line."""
        if (self.state != self.PRESENT or self.skeleton is None
                or now - self.last_body > SHOULDER_MAX_AGE):
            return None
        x, y, v = self.skeleton[POSE_NOSE]
        if v < SHOULDER_MIN_VIS:
            return None
        if self.nose_ema is None or now - self.nose_t > 1.0:
            self.nose_ema = y
        else:
            self.nose_ema += SHOULDER_SMOOTH * (y - self.nose_ema)
        self.nose_t = now
        return self.nose_ema

    def body_frame(self, now):
        """(shoulder_cx, shoulder_cy, shoulder_width) of YOU for the zoom, or None."""
        if (self.state != self.PRESENT or self.skeleton is None
                or now - self.last_body > SHOULDER_MAX_AGE):
            return None
        return shoulder_center(self.skeleton)

    def owns_hand(self, now, wrist):
        """Is this hand attached to YOU (and not to someone else standing nearby)?"""
        sk = self.skeleton
        if sk is None:
            return False
        for i in POSE_HAND_POINTS:               # sits on one of your own hand/wrist points
            x, y, v = sk[i]
            if v > 0.2 and math.hypot(x - wrist[0], y - wrist[1]) < HAND_OWN_DIST:
                return True
        sc = shoulder_center(sk)
        if sc is None:
            return False
        mine = math.hypot(wrist[0] - sc[0], wrist[1] - sc[1])
        if mine > HAND_REACH * sc[2]:            # too far from your shoulders to be your hand
            return False
        for t in self.ignored(now):              # closer to somebody else's shoulders -> theirs
            osc = shoulder_center(t.skel)
            if osc and math.hypot(wrist[0] - osc[0], wrist[1] - osc[1]) < mine:
                return False
        return True

    # -- identity tracking ----------------------------------------------------
    def _associate(self, now, detections):
        pairs = []
        for ti, t in enumerate(self.cands):
            gate = min(CAND_GATE_BASE + CAND_GATE_GROW * (now - t.last_seen), CAND_GATE_MAX)
            for di, (pos, _) in enumerate(detections):
                d = math.hypot(pos[0] - t.pos[0], pos[1] - t.pos[1])
                if d <= gate:
                    pairs.append((d, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for d, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self.cands[ti].update(detections[di][0], detections[di][1], now)
        # a detection no track was close enough to, but whose body box overlaps a person we lost
        # track of = the SAME person (their centre just jumped), not a new one
        for di, (pos, skel) in enumerate(detections):
            if di in used_d:
                continue
            box = skel_box(skel)
            if box is None:
                continue
            best, best_iou = None, REID_IOU
            for ti, t in enumerate(self.cands):
                if ti in used_t:
                    continue
                tb = skel_box(t.skel)
                if tb is not None:
                    iou = box_overlap(box, tb)[0]
                    if iou >= best_iou:
                        best, best_iou = ti, iou
            if best is not None:
                used_t.add(best)
                used_d.add(di)
                self.cands[best].update(pos, skel, now)
        for di, (pos, skel) in enumerate(detections):
            if di not in used_d:
                self.cands.append(CandTrack(pos, skel, now))
        self.cands = [t for t in self.cands if now - t.last_seen <= CAND_EXPIRE]
        # delete ghost tracks: not seen right now, sitting on top of someone who is
        seen_boxes = [skel_box(t.skel) for t in self.cands if t.last_seen == now]
        seen_boxes = [b for b in seen_boxes if b is not None]
        if seen_boxes:
            def ghost(t):
                if t.last_seen == now or t.id == self.target_id:
                    return False
                tb = skel_box(t.skel)
                return tb is not None and any(box_overlap(tb, sb)[0] >= 0.4 for sb in seen_boxes)
            self.cands = [t for t in self.cands if not ghost(t)]

    def _pick_target(self, now, hand_points):
        tgt = next((t for t in self.cands if t.id == self.target_id), None)
        if tgt is not None:
            return tgt                    # caller checks whether it was seen this frame
        if self.state != self.PRESENT:
            return None
        seen = [t for t in self.cands if t.last_seen == now]
        if not seen:
            return None
        if self.last_pos is None:
            seen = [t for t in seen if self._looks_like_you(t, now, hand_points)]
            if not seen:
                return None
            best = seen[0]
        else:
            gate = min(CAND_GATE_BASE + CAND_GATE_GROW * (now - self.last_body), CAND_GATE_MAX)
            best, bd = None, gate
            for t in seen:               # only something right where you were can be you
                d = math.hypot(t.pos[0] - self.last_pos[0], t.pos[1] - self.last_pos[1])
                if d <= bd:
                    best, bd = t, d
            if best is None:
                return None
        self.target_id = best.id
        return best

    # -- exit logic ---------------------------------------------------------------
    def _velocity(self):
        if len(self.motion) < 2:
            return 0.0, 0.0
        t_last = self.motion[-1][0]
        recent = [m for m in self.motion if t_last - m[0] <= 0.6]
        dt = recent[-1][0] - recent[0][0]
        if len(recent) < 2 or dt < 0.1:
            return 0.0, 0.0
        return (recent[-1][1] - recent[0][1]) / dt, (recent[-1][2] - recent[0][2]) / dt

    def _exit_edge(self):
        """Edge / 'door' you left through, or None if you vanished mid-view."""
        if self.last_pos is None:
            return None
        x, y = self.last_pos
        for x0, y0, x1, y1 in DOOR_ZONES:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return "door"
        vx, vy = self._velocity()
        edges = {"left": (x, -vx), "right": (1 - x, vx), "top": (y, -vy), "bottom": (1 - y, vy)}
        best, best_t = None, 1e9
        for name, (d, toward) in edges.items():
            if d < EDGE_HARD:
                t = 0.0
            elif toward > MIN_EXIT_SPEED and d / toward < EXIT_LOOKAHEAD:
                t = d / toward
            else:
                continue
            if t < best_t:
                best, best_t = name, t
        return best

    def _go_absent(self, now, expect_edge):
        self.state, self.state_since = self.ABSENT, now
        self.expect_edge = expect_edge
        self.target_id = None
        self.hands_since = None
        self.trail.clear()
        self.motion.clear()
        self.skeleton = None

    # -- entry logic ---------------------------------------------------------------
    @staticmethod
    def _entry_zone(t):
        """Which edge / doorway did this thing come in from? (None = it just showed up mid-view)"""
        for x0, y0, x1, y1 in DOOR_ZONES:
            if x0 <= t.first_pos[0] <= x1 and y0 <= t.first_pos[1] <= y1:
                return "door"
        edge = nearest_edge(t.first_pos)
        if edge:
            return edge
        dt = t.last_seen - t.first_seen          # spotted late? project its path backwards
        if dt >= 0.3:
            vx = (t.pos[0] - t.first_pos[0]) / dt
            vy = (t.pos[1] - t.first_pos[1]) / dt
            ox, oy = t.first_pos[0] - vx * ENTRY_BACKTRACK, t.first_pos[1] - vy * ENTRY_BACKTRACK
            if ox < 0.05:
                return "left"
            if ox > 0.95:
                return "right"
            if oy < 0.05:
                return "top"
            if oy > 0.95:
                return "bottom"
        return None

    def _zone_allowed(self, zone):
        if zone is None:
            return False
        if RETURN_SAME_SIDE_ONLY and self.expect_edge in ("left", "right", "top", "bottom", "door"):
            return zone == self.expect_edge
        return True

    @staticmethod
    def _moved_inward(t, zone):
        dx, dy = t.pos[0] - t.first_pos[0], t.pos[1] - t.first_pos[1]
        inward = {"left": dx, "right": -dx, "top": dy, "bottom": -dy}.get(zone)
        if inward is None:                         # doorway: any real movement
            return math.hypot(dx, dy) >= ACQUIRE_MIN_MOVE
        return inward >= ACQUIRE_MIN_MOVE

    @staticmethod
    def _looks_like_you(t, now, hand_points):
        """A body-shaped thing counts as a person if a hand is attached to it, or if it
        isn't a static lump and it either moved or has one of your hands right next to it."""
        if t.hand_linked(now):
            return True
        if t.is_static(now):
            return False
        if math.hypot(t.pos[0] - t.first_pos[0], t.pos[1] - t.first_pos[1]) >= ACQUIRE_MIN_MOVE:
            return True
        return any(math.hypot(hx - t.pos[0], hy - t.pos[1]) < HAND_NEAR for hx, hy in hand_points)

    def _accept(self, now, t, zone, silent=False):
        self.target_id = t.id
        self.state, self.state_since = self.PRESENT, now
        self.expect_edge = None
        self.last_pos, self.last_body, self.last_evidence = t.pos, now, now
        self.skeleton = t.skel
        self.trail.clear()
        self.motion.clear()
        return None if silent else ("Entered", zone)

    # -- chest claim: both hands on chest = this body is me -------------------------
    def _chest_claim(self, now, hand_points):
        best = None
        for t in self.cands:
            if t.last_seen != now:
                continue
            box = chest_box(t.skel)
            if box is None:
                continue
            x0, y0, x1, y1 = box

            def inside(x, y):
                return x0 <= x <= x1 and y0 <= y <= y1

            n_hands = sum(1 for hx, hy in hand_points if inside(hx, hy))
            n_pose = sum(1 for i in POSE_WRISTS
                         if t.skel[i][2] > 0.3 and inside(t.skel[i][0], t.skel[i][1]))
            if max(n_hands, n_pose) >= 2:
                best = t
                break

        if best is None:
            if now - self.chest_last > CHEST_GRACE:
                self.chest_id = None
            return None
        self.chest_last = now
        if self.chest_id != best.id:
            self.chest_id, self.chest_since = best.id, now
            return None
        if now - self.chest_since >= CHEST_HOLD and now - self.last_claim > CHEST_COOLDOWN:
            self.last_claim = now
            self.chest_id = None
            self.locked = LOCK_TO_CLAIMED
            if not (self.target_id == best.id and self.state == self.PRESENT):
                self._accept(now, best, None, silent=True)
            return ("Claimed", None)
        return None

    def _update_absent(self, now, hand_points):
        if not self.auto_ok:                      # several people in view: only the chest gesture picks you
            return None
        hands_seen = bool(hand_points)
        if HAND_LINK_ACCEPTS and not self.locked:                 # a body with a hand on it = you
            for t in self.cands:
                if t.last_seen == now and t.hand_linked(now):
                    silent = t.first_seen - self.start < STARTUP_SILENT
                    return self._accept(now, t, self._entry_zone(t), silent=silent)
        for t in self.cands:
            if t.last_seen != now or now - t.run_start < ENTER_CONFIRM:
                continue
            if t.first_seen - self.start < STARTUP_SILENT:       # in view when the script launched
                if self._looks_like_you(t, now, hand_points):
                    return self._accept(now, t, None, silent=True)
                continue
            if t.is_static(now):                                  # a lump: never counts
                continue
            zone = self._entry_zone(t)
            if self._zone_allowed(zone) and self._moved_inward(t, zone):
                return self._accept(now, t, zone)
            if (self.expect_edge is None and now - t.first_seen >= FALLBACK_SECONDS
                    and t.max_disp >= FALLBACK_MOVE):
                return self._accept(now, t, zone)

        # seated at a desk at launch: hands alone can establish presence
        if hands_seen and not self.locked and now - self.start < STARTUP_SILENT + ENTER_CONFIRM:
            if self.hands_since is None:
                self.hands_since = now
            if now - self.hands_since >= ENTER_CONFIRM:
                self.state, self.state_since = self.PRESENT, now
                self.last_evidence = now
                self.hands_since = None
        elif not hands_seen:
            self.hands_since = None
        return None

    # -- main update ----------------------------------------------------------------
    def update(self, now, detections, hand_points):
        """detections: [((x, y), skeleton), ...] for every body-shaped thing found this frame.
        Returns (kind, detail) on a state change:
        ("Entered", edge) / ("Left", edge) / ("Hidden", None) / ("Reappeared", None) / ("Claimed", None).
        hand_points: [(x, y), ...] wrist positions of ALL visible hands."""
        self._associate(now, dedupe_detections(detections))
        for t in self.cands:                      # which bodies have a visible hand attached?
            if t.last_seen == now:
                t.update_link(now, hand_points)
        people = [t for t in self.cands if now - t.last_seen < 0.5
                  and (t.id == self.target_id or not t.is_static(now) or t.hand_linked(now))]
        self.people_count = len(people)
        self.auto_ok = len(people) <= 1 or not AUTO_ID_ONLY_WHEN_ALONE

        target = self._pick_target(now, hand_points)
        body = target is not None and target.last_seen == now
        if body:
            self.last_pos, self.last_body, self.skeleton = target.pos, now, target.skel
            self.trail.append(target.pos)
            self.motion.append((now, target.pos[0], target.pos[1]))
        if self.skeleton is not None and self.state != self.ABSENT:
            hands_seen = any(self.owns_hand(now, hp) for hp in hand_points)   # only YOUR hands count
        else:
            hands_seen = bool(hand_points)
        evidence = body or hands_seen
        if evidence:
            self.last_evidence = now

        claim = self._chest_claim(now, hand_points)
        if claim:
            return claim

        if self.state == self.ABSENT:
            return self._update_absent(now, hand_points)

        if self.state == self.PRESENT:
            if not evidence and now - self.last_evidence >= DISAPPEAR_GRACE:
                edge = self._exit_edge()
                if edge:
                    self._go_absent(now, edge)
                    return ("Left", edge)
                self.state, self.state_since = self.HIDDEN, now
                return ("Hidden", None)
            return None

        # HIDDEN: stay "in the room"; a body with a hand on it, or something that MOVES
        # into view near where you hid, resumes tracking
        for t in self.cands:
            if t.last_seen != now:
                continue
            same_person = t.id == self.target_id
            if not same_person and t.first_seen < self.state_since - DISAPPEAR_GRACE - 1.0:
                continue                      # someone who was already in view isn't you coming back
            linked = (HAND_LINK_ACCEPTS and t.hand_linked(now) and self.auto_ok
                      and (same_person or not self.locked))
            if not linked:
                if now - t.run_start < RESUME_CONFIRM:
                    continue
                near = (math.hypot(t.pos[0] - self.last_pos[0], t.pos[1] - self.last_pos[1])
                        <= HIDDEN_RESUME_RADIUS)
                if not (near and not t.is_static(now) and t.run_move() >= ACQUIRE_MIN_MOVE):
                    continue
            self.target_id = t.id
            self.state, self.state_since = self.PRESENT, now
            self.last_pos, self.last_body = t.pos, now
            self.skeleton = t.skel
            return ("Reappeared", None)
        if HIDDEN_MAX and now - self.state_since >= HIDDEN_MAX:
            self._go_absent(now, None)
            return ("Left", "timeout")
        return None


def room_message(kind, detail):
    if kind == "Entered":
        if detail == "door":
            return "Entered the room (through the doorway)"
        return f"Entered the room (from {detail} edge)" if detail else "Entered the room"
    if kind == "Left":
        if detail == "timeout":
            return "Left the room (hidden too long)"
        if detail == "door":
            return "Left the room (through the doorway)"
        return f"Left the room (via {detail} edge)"
    if kind == "Hidden":
        return "Hidden - vanished mid-view, still counted as in the room"
    if kind == "Claimed":
        return "Hands on chest - this is YOU (others ignored)"
    return "Reappeared - tracking resumed"


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def to_px(pt, view, size):
    """Normalised FULL-frame point -> pixel inside a panel that shows `view` = (x0, y0, cw, ch)."""
    x0, y0, cw, ch = view
    return int((pt[0] - x0) / cw * size[0]), int((pt[1] - y0) / ch * size[1])


def draw_hand(canvas, pts, view, size, line_color=(0, 255, 0), dot_color=(0, 0, 255)):
    px = [to_px((p.x, p.y), view, size) for p in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, px[a], px[b], line_color, 2, cv2.LINE_AA)
    for p in px:
        cv2.circle(canvas, p, 4, dot_color, -1, cv2.LINE_AA)


def put_label(frame, text, x, y, color=(0, 255, 0), scale=0.8):
    h, w, _ = frame.shape
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = min(max(x, 5), max(w - tw - 10, 5))
    y = min(max(y, th + 10), h - 10)
    cv2.rectangle(frame, (x - 5, y - th - 8), (x + tw + 5, y + 8), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def match_track(tracks, wrist, used, now):
    """Find the existing track closest to this wrist position."""
    best, best_d = None, TRACK_MATCH_DIST
    for t in tracks:
        if t.id in used:
            continue
        d = math.hypot(t.last_pos[0] - wrist[0], t.last_pos[1] - wrist[1])
        if d < best_d:
            best, best_d = t, d
    return best


class GestureEngine:
    def __init__(self, is_awake, on_double_clap, on_peace, on_fist_close, on_swipe,
                 on_stop_speech=None):
        self.is_awake = is_awake
        self.on_double_clap = on_double_clap
        self.on_peace = on_peace
        self.on_fist_close = on_fist_close
        self.on_swipe = on_swipe
        self.on_stop_speech = on_stop_speech

        self.running = False
        self.error = None
        self._thread = None
        self._stop = threading.Event()
        self._viewers = 0
        self._vlock = threading.Lock()
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self._fist_progress = None
        self._open_progress = None

    # -- control ---------------------------------------------------------------
    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    # -- camera stream for the UI ----------------------------------------------
    def viewer_add(self, n):
        with self._vlock:
            self._viewers = max(0, self._viewers + n)

    def wait_frame(self, last_seq, timeout=2.0):
        """Blocks until a frame newer than last_seq exists. Returns (seq, jpeg) or (last_seq, None)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or not self.running, timeout)
            if self._seq != last_seq and self._jpeg is not None:
                return self._seq, self._jpeg
        return last_seq, None

    def _publish(self, img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with self._cond:
                self._jpeg = buf.tobytes()
                self._seq += 1
                self._cond.notify_all()

    # -- thread body -----------------------------------------------------------
    def _run(self):
        try:
            self._loop()
        except Exception as e:
            traceback.print_exc()
            self.error = str(e)
            print("[gesture] engine stopped:", e)
        finally:
            self.running = False
            with self._cond:
                self._cond.notify_all()

    def _loop(self):

        detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=ensure_model(MODEL_NAME, MODEL_URL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=HAND_MAX,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6))
        pose_detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=ensure_model(POSE_MODEL_NAME, POSE_MODEL_URL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=POSE_MAX_PEOPLE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4))

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (try CAMERA_INDEX = 1 in gesture_engine.py)")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
        base_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_FPS, base_fps * FPS_BOOST)

        audio = AudioClapListener()
        audio.start()
        print("[gesture] clap microphone check:", "ON" if audio.ok else f"OFF ({audio.error}) - camera only")
        decider = DoubleClapDecider()
        clap = ClapDetector()

        start = time.time()
        presence = PresenceTracker(start)
        zoom = ZoomView()
        tracks = []
        events = {}
        recent_allowed = []
        gestures_on = GESTURES_START_ON
        prev_awake = False
        peace_sleep = False      # Ada is going to sleep because of a peace sign
        fist_since = None        # when the current fist hold started
        fist_last = -1e9
        open_since = None        # when the current open-hand hold started
        open_last = -1e9
        open_fired = False       # fired already: the hand must close / leave before it can fire again
        first_clap_done = False  # the first double clap wakes Ada; later ones only toggle gestures
        frame_i = 0
        last_ts = -1
        last_clap = -1e9
        last_peace = -1e9
        prev_time = 0
        bad_reads = 0

        print("[gesture] engine running")
        try:
            while not self._stop.is_set():
                ok, raw = cap.read()
                if not ok:
                    bad_reads += 1
                    if bad_reads > 100:
                        raise RuntimeError("webcam stopped delivering frames")
                    time.sleep(0.05)
                    continue
                bad_reads = 0

                raw = cv2.flip(raw, 1)
                H, W = raw.shape[:2]
                w = PANEL_W
                h = int(w * H / W)
                frame = cv2.resize(raw, (w, h))
                now = time.time()
                ts = int((now - start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                awake = bool(self.is_awake())
                if prev_awake and not awake:      # Ada just went to sleep
                    if peace_sleep:               # ...by peace sign: gestures stay ON
                        peace_sleep = False
                    else:                         # ...by voice / double clap: gestures switch off too
                        gestures_on = False
                        tracks = []
                elif awake:
                    peace_sleep = False
                prev_awake = awake

                # hands: detected on the zoomed crop
                zx, zy, zw, zh = zoom.rect()
                px0, py0 = int(zx * W), int(zy * H)
                px1 = min(max(int((zx + zw) * W), px0 + 32), W)
                py1 = min(max(int((zy + zh) * H), py0 + 32), H)
                view = (px0 / W, py0 / H, (px1 - px0) / W, (py1 - py0) / H)
                crop = raw[py0:py1, px0:px1]
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), ts)
                hands_full = [[P(view[0] + l.x * view[2], view[1] + l.y * view[3]) for l in lms]
                              for lms in result.hand_landmarks]

                # bodies: always the full view
                detections = []
                frame_i += 1
                if frame_i % POSE_EVERY_N == 0:
                    pose_res = pose_detector.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
                    for pl in pose_res.pose_landmarks:
                        c = body_center(pl, tracking=False)
                        if c is None:
                            loose = body_center(pl, tracking=True)
                            c = loose if presence.near_target(loose) else None
                        if c is not None:
                            detections.append((c, skeleton_from(pl)))

                room_ev = presence.update(now, detections,
                                          [(hl[WRIST].x, hl[WRIST].y) for hl in hands_full])
                if room_ev:
                    kind, detail = room_ev
                    events["room"] = (room_message(kind, detail), now + ROOM_EVENT_DISPLAY)
                    print("[gesture] room:", room_message(kind, detail))

                zoom.update(now, presence.body_frame(now))
                shoulder_y = presence.nose_line(now)   # (variable name kept; it is the NOSE line now)

                # which hands are YOURS and above your shoulders
                hand_draws = []
                mine = []
                recent_allowed = [r for r in recent_allowed if now - r[0] < STICKY_TIME]
                for lms, handedness in zip(hands_full, result.handedness):
                    wrist = (lms[WRIST].x, lms[WRIST].y)
                    palm_y = palm_center(lms)[1]
                    if shoulder_y is None:
                        allowed = GESTURE_IF_NO_SHOULDERS
                    else:
                        sticky = any(math.hypot(rx - wrist[0], ry - wrist[1]) < 0.15
                                     for _, rx, ry in recent_allowed)
                        margin = GESTURE_MARGIN_STICKY if sticky else GESTURE_MARGIN
                        allowed = palm_y < shoulder_y + margin and presence.owns_hand(now, wrist)
                    if not allowed:
                        hand_draws.append((lms, (120, 120, 120), (120, 120, 120), None, None))
                        continue
                    mine.append((lms, handedness, wrist))
                    recent_allowed.append((now, wrist[0], wrist[1]))
                active_hands = [m[0] for m in mine]

                # ---- double clap: wake / sleep toggle (works any time) ----
                clap.update(now, active_hands)
                if decider.check(now, clap, audio):
                    last_clap = now
                    tracks = []
                    gestures_on = not gestures_on
                    events["both"] = ("Gestures ON" if gestures_on else "Gestures OFF",
                                      now + EVENT_DISPLAY * 2)
                    print("[gesture] double clap -> gestures", "ON" if gestures_on else "OFF")
                    if not first_clap_done:
                        first_clap_done = True
                        self.on_double_clap()
                locked_out = now - last_clap < LOCKOUT_AFTER_CLAP

                # ---- per-hand gestures ----
                used = set()
                for lms, handedness, wrist in mine:
                    if not gestures_on:               # identification only: no gesture reading
                        hand_draws.append((lms, (255, 200, 0), (255, 200, 0), None, None))
                        continue
                    track = match_track(tracks, wrist, used, now)
                    if track is None:
                        track = HandTrack(handedness[0].category_name, wrist, now)
                        tracks.append(track)
                    used.add(track.id)
                    live, changed, event = track.update(now, lms)

                    if not locked_out:
                        if (changed and live == "Peace"
                                and now - last_peace > PEACE_COOLDOWN):
                            last_peace = now
                            wake = not awake
                            peace_sleep = not wake
                            events[track.id] = ("Peace - wake" if wake else "Peace - sleep",
                                                now + EVENT_DISPLAY)
                            print("[gesture] peace sign ->", "wake" if wake else "sleep")
                            self.on_peace(wake)
                        if event != "Fist Close":
                            # a swipe = turning the open palm (either way) or moving it (any direction).
                            # WHICH hand did it decides the tab direction.
                            swiped = event in ("Swipe Right", "Swipe Left")
                            if not swiped and SWIPE_ANY_MOVE:
                                swiped = self._move_swipe(track, lms, now)
                            if swiped:
                                hand = handedness[0].category_name
                                if SWAP_HANDS:
                                    hand = "Left" if hand == "Right" else "Right"
                                direction = -1 if hand == "Right" else 1   # right hand = left, left hand = right
                                events[track.id] = (f"{hand} hand swipe -> "
                                                    f"{'next' if direction > 0 else 'previous'} tab",
                                                    now + EVENT_DISPLAY)
                                print("[gesture]", events[track.id][0])
                                self.on_swipe(direction)
                                open_since = None

                    tag = f"{track.name}: {live or '...'}"
                    ev = events[track.id][0] if track.id in events and events[track.id][1] > now else None
                    hand_draws.append((lms, (0, 255, 0), (0, 0, 255), tag, ev))
                # ---- hold a fist to shut down ----
                fist_now = (gestures_on and not locked_out
                            and any(t.shape == "Fist" and now - t.last_seen < 0.2 for t in tracks))
                if fist_now:
                    fist_last = now
                    if fist_since is None:
                        fist_since = now
                    elif now - fist_since >= FIST_HOLD_SECONDS:
                        fist_since = None
                        print("[gesture] fist held -> shutdown")
                        self.on_fist_close()
                elif fist_since is not None and now - fist_last > FIST_GRACE:
                    fist_since = None
                self._fist_progress = (now - fist_since) if fist_since is not None else None

                # ---- hold an open hand to stop Ada talking ----
                open_now = (gestures_on and not locked_out
                            and any(t.shape == "Open Hand" and now - t.last_seen < 0.2 for t in tracks))
                if open_now:
                    open_last = now
                    if open_since is None and not open_fired:
                        open_since = now
                    elif open_since is not None and now - open_since >= OPEN_HOLD_SECONDS:
                        open_since = None
                        open_fired = True
                        print("[gesture] open hand held -> stop speech")
                        if self.on_stop_speech:
                            self.on_stop_speech()
                elif now - open_last > OPEN_GRACE:
                    open_since = None
                    open_fired = False
                self._open_progress = (now - open_since) if open_since is not None else None

                tracks = [t for t in tracks if now - t.last_seen <= TRACK_GRACE]

                # ---- drawing (only when someone is watching) ----
                with self._vlock:
                    watching = self._viewers > 0
                if watching or SHOW_WINDOW:
                    out = self._draw(frame, crop, view, zoom, presence, hand_draws, events,
                                     shoulder_y, gestures_on, now, w, h, prev_time)
                    if watching:
                        self._publish(out)
                    if SHOW_WINDOW:
                        cv2.imshow("Ada gestures", out)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                prev_time = now
        finally:
            audio.stop()
            cap.release()
            if SHOW_WINDOW:
                cv2.destroyAllWindows()
            detector.close()
            pose_detector.close()

    # -- open-palm swipe in ANY direction ------------------------------------------
    @staticmethod
    def _move_swipe(track, lms, now):
        if track.shape != "Open Hand":
            track.swipe_hist.clear()
            return False
        cx, cy = palm_center(lms)
        track.swipe_hist.append((now, cx, cy, hand_size(lms)))
        while track.swipe_hist and now - track.swipe_hist[0][0] > SWIPE_WINDOW:
            track.swipe_hist.popleft()
        if len(track.swipe_hist) < SWIPE_MIN_SAMPLES:
            return False
        _, x0, y0, _ = track.swipe_hist[0]
        size = sum(sz for _, _, _, sz in track.swipe_hist) / len(track.swipe_hist)
        if math.hypot(cx - x0, cy - y0) < SWIPE_MOVE_MIN * size:
            return False
        if now - track.last_swipe < SWIPE_MOVE_COOLDOWN:
            return False
        track.last_swipe = now
        track.swipe_hist.clear()
        return True

    # -- drawing -----------------------------------------------------------------
    def _draw(self, frame, crop, view, zoom, presence, hand_draws, events,
              shoulder_y, gestures_on, now, w, h, prev_time):
        panels = [(frame, (0.0, 0.0, 1.0, 1.0))]
        zoom_panel = cv2.resize(crop, (w, h)) if STREAM_ZOOM_PANEL else None
        if zoom_panel is not None:
            panels.append((zoom_panel, view))

        for canvas, v in panels:
            for lms, lc, dc, tag, ev in hand_draws:
                draw_hand(canvas, lms, v, (w, h), lc, dc)
                if tag:
                    x, y = to_px((lms[WRIST].x, lms[WRIST].y), v, (w, h))
                    put_label(canvas, tag, x, y + 40)
                    if ev:
                        put_label(canvas, f">> {ev}", x, y + 80, (0, 200, 255))
            if shoulder_y is not None:
                ly = to_px((0, shoulder_y), v, (w, h))[1]
                if 0 <= ly < h:
                    cv2.line(canvas, (0, ly), (w, ly), (255, 200, 0), 1, cv2.LINE_AA)

        if "both" in events and events["both"][1] > now:
            put_label(frame, f">> Double clap: {events['both'][0]}", 10, h - 50, (255, 100, 255), 1.0)

        for t in presence.ignored(now):
            draw_skeleton(frame, t.skel, (60, 60, 255))
            gx, gy = int(t.pos[0] * w), int(t.pos[1] * h)
            tag = f"Person {t.id} - not you"
            if presence.chest_id == t.id and now - presence.chest_last < CHEST_GRACE:
                pct = min(int(100 * (now - presence.chest_since) / CHEST_HOLD), 99)
                tag = f"Person {t.id} - claiming {pct}%"
            put_label(frame, tag, gx - 70, gy - 22, (90, 90, 255), 0.5)

        if (presence.state == PresenceTracker.PRESENT and presence.skeleton
                and now - presence.last_body < 0.6):
            draw_skeleton(frame, presence.skeleton, (255, 200, 0))
            if presence.last_pos:
                put_label(frame, "YOU", int(presence.last_pos[0] * w) - 20,
                            int(presence.last_pos[1] * h) - 40, (255, 200, 0), 0.6)

        if zoom.z > 1.02:
            p0 = to_px((view[0], view[1]), (0.0, 0.0, 1.0, 1.0), (w, h))
            p1 = to_px((view[0] + view[2], view[1] + view[3]), (0.0, 0.0, 1.0, 1.0), (w, h))
            cv2.rectangle(frame, p0, p1, (255, 255, 0), 1, cv2.LINE_AA)

        elapsed = fmt_time(now - presence.state_since)
        if presence.state == PresenceTracker.PRESENT:
            status, colour = f"IN ROOM  {elapsed}", (0, 255, 0)
        elif presence.state == PresenceTracker.HIDDEN:
            status, colour = f"HIDDEN  {elapsed}", (0, 215, 255)
        else:
            status, colour = f"AWAY  {elapsed}", (0, 165, 255)
        if presence.people_count > 1:
            status += f"  |  {presence.people_count} people"
        put_label(frame, status, 10, 60, colour, 0.6)
        if presence.people_count > 1 and presence.state != PresenceTracker.PRESENT:
            put_label(frame, "Put BOTH hands on your chest to take control",
                        10, 180, (0, 200, 255), 0.55)
        if "room" in events and events["room"][1] > now:
            put_label(frame, f">> {events['room'][0]}", 10, 100, (0, 255, 255), 0.6)
        if self._open_progress is not None and self._open_progress > 1.0:
            left = max(OPEN_HOLD_SECONDS - self._open_progress, 0)
            msg = f"HOLD OPEN HAND: stopping Ada in {left:.1f}s"
            put_label(frame, msg, 10, h - 125, (255, 200, 0), 0.6)
        if self._fist_progress is not None and self._fist_progress > 0.5:
            left = max(FIST_HOLD_SECONDS - self._fist_progress, 0)
            put_label(frame, f"HOLD FIST: shutting down in {left:.1f}s - open hand to cancel",
                        10, h - 90, (60, 60, 255), 0.6)
        if gestures_on:
            put_label(frame, "GESTURES: ON  (double clap = off | peace = sleep/wake)", 10, 140, (0, 255, 0), 0.55)
        else:
            put_label(frame, "GESTURES: OFF  (double clap = on)", 10, 140, (0, 165, 255), 0.55)

        fps = 1 / (now - prev_time) if prev_time else 0
        cv2.putText(frame, f"FPS {int(fps)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.hconcat([frame, zoom_panel]) if zoom_panel is not None else frame