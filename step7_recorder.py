# ===================== STEP 7: DATA RECORDER =====================
# Runs a fixed, cue-based protocol and saves the raw measurements of EVERY
# frame to a CSV file. No video is saved: only numbers.
# Later, the baseline and the improved detector both read the SAME CSV file,
# so they can be compared fairly.

# ===================== IMPORTS =====================
import csv                                        # write CSV files
import math                                       # atan2, degrees (head tilt)
import os                                         # make folders, build file paths
import subprocess                                 # play a short beep sound on the Mac
import time                                       # clock: timestamps
from datetime import datetime                     # date + time for the file name

import cv2                                        # OpenCV: camera, drawing, window
import numpy as np                                # NumPy: math (distances, matrices)
import mediapipe as mp                            # MediaPipe
from mediapipe.tasks import python as mp_python   # MediaPipe task settings (BaseOptions)
from mediapipe.tasks.python import vision         # MediaPipe face landmarker

# ===================== SETTINGS =====================
CAMERA = 1                                        # camera number (1 = Mac built-in camera on my Mac)
MODEL_PATH = "models/face_landmarker.task"        # the downloaded AI model
DATA_FOLDER = "data"                              # where the CSV files go

NATURAL_SECONDS = 20                             # natural blinking phase: 3 minutes
N_LONG_BLINKS = 2                                # how many "LONG BLINK" cues (professor: 10-15)
N_LOOK_DOWNS = 2                                 # how many "LOOK DOWN" cues (professor: ~10)
WAIT_SECONDS = 4                                  # relaxed pause before each cue
CUE_SECONDS = 3                                   # how long each cue stays on screen

RIGHT_EYE = [33, 160, 158, 133, 153, 144]         # p1..p6 of the right eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]         # p1..p6 of the left eye

# NEW: raw points saved for later experiments (16-point EAR, iris position)
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_IRIS = [468, 469, 470, 471, 472]            # iris centre (468) + 4 points around it
LEFT_IRIS = [473, 474, 475, 476, 477]             # iris centre (473) + 4 points around it
RAW_POINTS = RIGHT_EYE_CONTOUR + LEFT_EYE_CONTOUR + RIGHT_IRIS + LEFT_IRIS   # 42 points


# ===================== FUNCTIONS =====================
def get_eye_points(face, eye, w, h):
    """Turn MediaPipe's 6 eye landmarks into 6 (x, y) pixel positions."""
    points = []
    for i in eye:
        points.append([face[i].x * w, face[i].y * h])
    return np.array(points)


def eye_aspect_ratio(pts):
    """EAR formula from Soukupova & Cech (2016)."""
    p1, p2, p3, p4, p5, p6 = pts
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    return vertical / horizontal


def head_pitch(matrix):
    """Up/down tilt of the head in degrees, from MediaPipe's 4x4 head-pose matrix.
    The sign (+ = up or down?) is checked in the pilot recording."""
    r = np.array(matrix)[:3, :3]                  # the rotation part of the matrix
    return math.degrees(math.atan2(r[2, 1], r[2, 2]))


def draw_text(frame, text, pos, scale=0.8, color=(255, 255, 255)):
    """Text with a black outline, readable on any background."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def beep():
    """Short sound at each new cue (macOS). If it fails, just continue silently."""
    try:
        subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])
    except Exception:
        pass


def build_protocol():
    """The list of steps every participant goes through, in order.
    Each step = (name, label, seconds, instruction shown on screen)."""
    steps = []
    steps.append(("rest", "none", 10, "Get comfortable and look at the screen"))
    steps.append(("natural", "natural", NATURAL_SECONDS, "Relax, look at the screen, blink normally"))
    for _ in range(N_LONG_BLINKS):
        steps.append(("wait", "natural", WAIT_SECONDS, "Relax..."))
        steps.append(("cue_blink", "intentional", CUE_SECONDS, "LONG BLINK NOW"))
    for _ in range(N_LOOK_DOWNS):
        steps.append(("wait", "natural", WAIT_SECONDS, "Look at the screen"))
        steps.append(("cue_down", "lookdown", CUE_SECONDS, "LOOK DOWN at the keyboard, count to 3, look up"))
    steps.append(("end", "none", 3, "Done - thank you!"))
    return steps


def current_step(steps, elapsed):
    """Which step are we in, 'elapsed' seconds after the start? Returns (index, seconds left)."""
    total = 0.0
    for index, step in enumerate(steps):
        total = total + step[2]                   # add this step's duration
        if elapsed < total:
            return index, total - elapsed
    return None, 0.0                              # protocol finished


# ===================== ASK WHO AND UNDER WHICH CONDITION =====================
participant = input("Participant ID (e.g. P01): ").strip().replace(" ", "_")
condition = input("Condition (e.g. normal_light, glasses, dim_light): ").strip().replace(" ", "_")

os.makedirs(DATA_FOLDER, exist_ok=True)           # create the data folder if it doesn't exist
stamp = datetime.now().strftime("%Y%m%d_%H%M")    # e.g. 20261106_1530
csv_path = os.path.join(DATA_FOLDER, f"{participant}_{condition}_{stamp}_frames.csv")

# ===================== FACE FINDER (now also with blendshapes + head pose) =====================
options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,                 # NEW: scores like eyeBlink and eyeLookDown
    output_facial_transformation_matrixes=True,   # NEW: head pose (for head tilt)
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ===================== CAMERA =====================
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    print("Camera could not be opened")
    exit()

# ===================== CSV FILE: one row per frame =====================
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["participant", "condition", "t", "step_index", "step_name", "label", "face",
                 "ear_left", "ear_right", "ear", "blink_left", "blink_right",
                 "lookdown_left", "lookdown_right", "pitch", "img_w", "img_h"]
                + [f"{axis}{i}" for i in RAW_POINTS for axis in ("x", "y")])   # NEW: x33, y33, x7, y7, ...

steps = build_protocol()
started = False                                   # becomes True when SPACE is pressed
protocol_start = 0.0
last_step_index = -1
fails = 0
start = time.time()

print("Press SPACE in the camera window to start the recording, q to stop.")

# ===================== MAIN LOOP =====================
while True:
    ok, frame = cap.read()
    if not ok:
        fails = fails + 1
        if fails > 100:
            print("Camera never started")
            break
        continue
    fails = 0
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    now = time.time()

    # ----- MediaPipe -----
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))

    # ----- Measurements (empty if no face) -----
    face_found = 1 if result.face_landmarks else 0
    values = [""] * (8 + 2 + 2 * len(RAW_POINTS))  # all empty: 8 measurements + image size + 42 points × (x, y)
    if face_found:
        face = result.face_landmarks[0]
        ear_l = eye_aspect_ratio(get_eye_points(face, LEFT_EYE, w, h))
        ear_r = eye_aspect_ratio(get_eye_points(face, RIGHT_EYE, w, h))
        scores = {}                               # blendshape name -> score (0..1)
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                scores[category.category_name] = category.score
        pitch = ""
        if result.facial_transformation_matrixes:
            pitch = round(head_pitch(result.facial_transformation_matrixes[0]), 2)
        values = [round(ear_l, 4), round(ear_r, 4), round((ear_l + ear_r) / 2, 4),
                  round(scores.get("eyeBlinkLeft", 0), 4), round(scores.get("eyeBlinkRight", 0), 4),
                  round(scores.get("eyeLookDownLeft", 0), 4), round(scores.get("eyeLookDownRight", 0), 4),
                  pitch, w, h]
        for i in RAW_POINTS:                      # NEW: pixel position of every eye + iris point
            values.append(round(face[i].x * w, 1))
            values.append(round(face[i].y * h, 1))
        for i in RIGHT_EYE + LEFT_EYE:            # eye dots, so we can see tracking works
            cv2.circle(frame, (int(face[i].x * w), int(face[i].y * h)), 2, (0, 255, 0), -1)

    # ----- Protocol: which step, and save the row -----
    if started:
        index, left = current_step(steps, now - protocol_start)
        if index is None:                         # all steps done
            break
        name, label, _, instruction = steps[index]
        if index != last_step_index:              # a new step just began
            last_step_index = index
            if name.startswith("cue"):
                beep()
        writer.writerow([participant, condition, round(now - protocol_start, 3), index, name, label,
                         face_found] + values)
        color = (0, 255, 255) if name.startswith("cue") else (255, 255, 255)
        draw_text(frame, instruction, (40, h // 2), 1.3, color)
        draw_text(frame, f"Step {index + 1}/{len(steps)}   {left:.0f} s", (40, h - 40), 0.7)
    else:
        draw_text(frame, "Press SPACE to start", (40, h // 2), 1.3, (0, 255, 255))

    if not face_found:
        draw_text(frame, "No face", (40, 50), 1, (0, 0, 255))
    else:
        draw_text(frame, f"EAR {values[2]}   look-down {values[5]}   pitch {values[7]}", (40, 50), 0.7)

    cv2.imshow("BlinkTalk recorder", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") and not started:           # SPACE starts the protocol
        started = True
        protocol_start = time.time()
    if key == ord("q"):                           # q stops early (what was recorded is kept)
        break

# ===================== CLEANUP =====================
csv_file.close()                                  # make sure everything is written to disk
cap.release()
landmarker.close()
cv2.destroyAllWindows()
print(f"Saved: {csv_path}")