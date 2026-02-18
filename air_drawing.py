"""
╔══════════════════════════════════════════════════════════════╗
║           AIR DRAWING SYSTEM - Full Featured                 ║
║     Compatible with MediaPipe 0.10+ (new API)               ║
║                                                              ║
║  GESTURES:                                                   ║
║  ✦ Index finger only UP    → DRAW                           ║
║  ✦ Index + Middle UP       → IDLE / Move cursor             ║
║  ✦ All 4 fingers UP        → ERASE                          ║
║                                                              ║
║  KEYBOARD:                                                   ║
║  C  → Clear canvas                                          ║
║  S  → Save drawing as PNG                                   ║
║  U  → Undo last stroke                                      ║
║  ESC → Quit                                                  ║
╚══════════════════════════════════════════════════════════════╝


"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CAMERA_INDEX     = 0      # Try 1 or 2 if camera not found
FRAME_WIDTH      = 1280
FRAME_HEIGHT     = 720
SMOOTHING_WINDOW = 5
MAX_UNDO_STEPS   = 20

# Colors in BGR format
COLORS = {
    "Blue":   (255, 150, 50),
    "Green":  (50,  220, 50),
    "Red":    (50,  50,  255),
    "Yellow": (0,   220, 220),
    "White":  (255, 255, 255),
    "Purple": (255, 50,  200),
    "Orange": (0,   140, 255),
}

# ─────────────────────────────────────────────
#  MEDIAPIPE 0.10+ SETUP
# ─────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult  = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode     = mp.tasks.vision.RunningMode

# Hand connections for skeleton drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Download model if not present
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe hand model (~9MB)... please wait")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded!")

# Shared result from async callback
class SharedResult:
    def __init__(self):
        self.result = None

latest = SharedResult()

def result_callback(result: HandLandmarkerResult, output_image, timestamp_ms: int):
    latest.result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    result_callback=result_callback,
)
landmarker = HandLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
#  GESTURE DETECTION
# ─────────────────────────────────────────────
def fingers_up(landmarks, handedness_label):
    tips = [4, 8, 12, 16, 20]
    pip  = [3, 6, 10, 14, 18]
    up   = []
    is_right = (handedness_label == "Right")
    if is_right:
        up.append(landmarks[tips[0]].x < landmarks[pip[0]].x)
    else:
        up.append(landmarks[tips[0]].x > landmarks[pip[0]].x)
    for i in range(1, 5):
        up.append(landmarks[tips[i]].y < landmarks[pip[i]].y)
    return up

def get_mode(fingers):
    _, index, middle, ring, pinky = fingers
    if index and not middle and not ring and not pinky:
        return "DRAW"
    if index and middle and not ring and not pinky:
        return "IDLE"
    if index and middle and ring and pinky:
        return "ERASE"
    return "IDLE"

# ─────────────────────────────────────────────
#  DRAWING STATE
# ─────────────────────────────────────────────
class DrawingState:
    def __init__(self, w, h):
        self.canvas      = np.zeros((h, w, 3), dtype=np.uint8)
        self.color_name  = "Blue"
        self.color       = COLORS["Blue"]
        self.line_thick  = 5
        self.eraser_size = 60
        self.mode        = "IDLE"
        self.prev_point  = None
        self.smooth_pts  = deque(maxlen=SMOOTHING_WINDOW)
        self.undo_stack  = deque(maxlen=MAX_UNDO_STEPS)
        self.fps_buf     = deque(maxlen=30)
        self.last_t      = time.time()
        self.save_flash  = 0
        self.undo_flash  = 0

    def push_undo(self):
        self.undo_stack.append(self.canvas.copy())

    def undo(self):
        if self.undo_stack:
            self.canvas = self.undo_stack.pop()
            self.prev_point = None
            self.undo_flash = 40

    def clear(self):
        self.push_undo()
        self.canvas[:] = 0
        self.prev_point = None

    def save(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"air_drawing_{ts}.png"
        cv2.imwrite(name, self.canvas)
        self.save_flash = 60
        print(f"Saved: {name}")
        return name

    def smooth(self, pt):
        self.smooth_pts.append(pt)
        return (int(np.mean([p[0] for p in self.smooth_pts])),
                int(np.mean([p[1] for p in self.smooth_pts])))

    def fps(self):
        now = time.time()
        self.fps_buf.append(1.0 / max(now - self.last_t, 1e-6))
        self.last_t = now
        return int(np.mean(self.fps_buf))

# ─────────────────────────────────────────────
#  UI BUTTONS
# ─────────────────────────────────────────────
UI_H = 80

def build_buttons(w):
    btns = []
    x = 10
    for name, bgr in COLORS.items():
        btns.append({"label": name, "x1": x, "y1": 8, "x2": x+78, "y2": 70,
                     "action": "color", "value": name, "bgr": bgr})
        x += 82
    x += 8
    for sym, delta in [("-", -1), ("+", 1)]:
        btns.append({"label": f"L{sym}", "x1": x, "y1": 8, "x2": x+44, "y2": 70,
                     "action": "line", "value": delta, "bgr": (70, 70, 70)})
        x += 48
    x += 8
    for sym, delta in [("-", -5), ("+", 5)]:
        btns.append({"label": f"E{sym}", "x1": x, "y1": 8, "x2": x+44, "y2": 70,
                     "action": "eraser", "value": delta, "bgr": (55, 55, 55)})
        x += 48
    x += 8
    for lbl, act in [("CLEAR", "clear"), ("SAVE", "save"), ("UNDO", "undo")]:
        btns.append({"label": lbl, "x1": x, "y1": 8, "x2": x+72, "y2": 70,
                     "action": act, "value": None, "bgr": (35, 35, 35)})
        x += 76
    return btns

def draw_ui(frame, state, buttons, fps):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, UI_H), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)

    for b in buttons:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        active = (b["action"] == "color" and b["value"] == state.color_name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), b["bgr"], -1)
        border = (255, 255, 255) if active else (100, 100, 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2 if active else 1)
        lbl = b["label"]
        fs = 0.42
        tw, th = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
        cv2.putText(frame, lbl,
                    (x1 + (x2-x1-tw)//2, y1 + (y2-y1+th)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

    sx = w - 390
    mc = {"DRAW": (0,220,0), "ERASE": (50,80,255), "IDLE": (170,170,170)}.get(state.mode, (180,180,180))
    cv2.putText(frame, f"MODE: {state.mode}", (sx, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, mc, 2, cv2.LINE_AA)
    cv2.putText(frame, f"LINE:{state.line_thick}  ERASER:{state.eraser_size}  FPS:{fps}",
                (sx, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190,190,190), 1, cv2.LINE_AA)

    cv2.putText(frame,
                "1 finger=DRAW  2 fingers=IDLE  4 fingers=ERASE  |  C=Clear  S=Save  U=Undo  ESC=Quit",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,120), 1, cv2.LINE_AA)

    if state.save_flash > 0:
        msg = "  Drawing Saved!  "
        tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0][0]
        cv2.putText(frame, msg, ((w-tw)//2, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 100), 3, cv2.LINE_AA)
        state.save_flash -= 1

    if state.undo_flash > 0:
        msg = "Undo!"
        tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
        cv2.putText(frame, msg, ((w-tw)//2, h//2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,200,0), 2, cv2.LINE_AA)
        state.undo_flash -= 1

def handle_button(cx, cy, buttons, state):
    if cy >= UI_H:
        return False
    for b in buttons:
        if b["x1"] <= cx <= b["x2"] and b["y1"] <= cy <= b["y2"]:
            a = b["action"]
            if a == "color":
                state.color_name = b["value"]
                state.color = COLORS[b["value"]]
            elif a == "line":
                state.line_thick = max(1, min(30, state.line_thick + b["value"]))
            elif a == "eraser":
                state.eraser_size = max(10, min(150, state.eraser_size + b["value"]))
            elif a == "clear":
                state.clear()
            elif a == "save":
                state.save()
            elif a == "undo":
                state.undo()
            return True
    return False

# ─────────────────────────────────────────────
#  SKELETON + CURSOR
# ─────────────────────────────────────────────
def draw_skeleton(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 80, 80), 1, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        color = (0, 200, 255) if i in [4, 8, 12, 16, 20] else (200, 200, 200)
        cv2.circle(frame, pt, 4, color, -1)

def draw_cursor(frame, pt, state):
    if state.mode == "DRAW":
        cv2.circle(frame, pt, state.line_thick // 2 + 5, state.color, -1)
        cv2.circle(frame, pt, state.line_thick // 2 + 7, (255,255,255), 1)
    elif state.mode == "ERASE":
        cv2.circle(frame, pt, state.eraser_size // 2, (80, 80, 255), 2)
    else:
        cv2.circle(frame, pt, 8, (200, 200, 200), 2)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Cannot open camera. Change CAMERA_INDEX at top of file.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}x{h}")

    state        = DrawingState(w, h)
    buttons      = build_buttons(w)
    btn_cooldown = 0
    ts_ms        = 0

    print("Air Drawing running! Use hand gestures to draw.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        ts_ms += 33

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarker.detect_async(mp_image, ts_ms)

        result = latest.result

        if result and result.hand_landmarks:
            landmarks  = result.hand_landmarks[0]
            handedness = result.handedness[0][0].display_name

            draw_skeleton(frame, landmarks, w, h)

            tip = landmarks[8]
            cx  = max(0, min(w-1, int(tip.x * w)))
            cy  = max(0, min(h-1, int(tip.y * h)))
            spt = state.smooth((cx, cy))

            fingers = fingers_up(landmarks, handedness)
            mode    = get_mode(fingers)

            if cy < UI_H and btn_cooldown == 0:
                if handle_button(cx, cy, buttons, state):
                    btn_cooldown = 20
                mode = "IDLE"

            if mode == "DRAW" and state.mode != "DRAW":
                state.push_undo()
                state.prev_point = None

            state.mode = mode

            if cy >= UI_H:
                if mode == "DRAW":
                    if state.prev_point:
                        cv2.line(state.canvas, state.prev_point, spt,
                                 state.color, state.line_thick, cv2.LINE_AA)
                    state.prev_point = spt
                elif mode == "ERASE":
                    cv2.circle(state.canvas, spt, state.eraser_size // 2, (0,0,0), -1)
                    state.prev_point = None
                else:
                    state.prev_point = None

            draw_cursor(frame, spt, state)

        else:
            state.prev_point = None
            state.smooth_pts.clear()
            state.mode = "IDLE"

        if btn_cooldown > 0:
            btn_cooldown -= 1

        # Blend canvas onto frame
        gray = cv2.cvtColor(state.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        blended = cv2.addWeighted(frame, 0.15, state.canvas, 0.85, 0)
        frame[mask > 0] = blended[mask > 0]

        draw_ui(frame, state, buttons, state.fps())
        cv2.imshow("AIR DRAWING  |  ESC to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('c'), ord('C')):
            state.clear()
        elif key in (ord('s'), ord('S')):
            state.save()
        elif key in (ord('u'), ord('U')):
            state.undo()

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")

if __name__ == "__main__":
    main()
