# BlinkTalk

Assistive communication from eye blinks using a normal webcam.
Project laboratory, BME (Budapest University of Technology and Economics), 2026.
Author: Bhuwaneshwaree Ragavan · Supervisor: Professor Al-Radhi Mohammed Salah Hamza

## Pipeline

Webcam → MediaPipe Face Landmarker → eye landmarks → Eye Aspect Ratio (EAR)
→ OPEN / CLOSED → blink detection + duration → natural vs. long blink → message

EAR is computed as in Soukupová & Čech (2016), *Real-Time Eye Blink Detection using Facial Landmarks*:

EAR = (|p2 − p6| + |p3 − p5|) / (2 · |p1 − p4|)

## Files (each step builds on the previous one)

| File | What it does |
|---|---|
| `step2_camera.py` | Opens the webcam and shows the image |
| `step4a_eyes.py` | MediaPipe finds the face; draws the 6 landmarks of each eye |
| `step4b_ear.py` | Computes and displays EAR for each eye |
| `step4c_state.py` | Threshold → eye OPEN / CLOSED |
| `step4d_blinks.py` | Counts blinks and measures each blink's duration |
| `step5_hello.py` | Long blink (≥ 0.4 s) → displays "Hello" |

## Setup (macOS)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir models
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
python step5_hello.py      # press q to quit
```

## First results (my own webcam)

| Run | Blinks | Natural | Long | Notes |
|---|---|---|---|---|
| 1 | 25 | 0.07–0.32 s | 0.53–0.97 s | clear gap between the groups |
| 2 | 13 | 0.10–0.33 s | 0.53–0.90 s | one long blink split into 3 parts |
| 3 | 30 | 0.07–0.40 s | 0.83–1.23 s | rule "≥ 0.4 s = long": 30/30 correct |

## Observations / open issues

1. Looking down lowers EAR and is wrongly detected as CLOSED.
2. A gently closed long blink keeps EAR close to the threshold, so it can be split into several blinks. Planned fix: two thresholds (hysteresis).
3. Open-eye EAR changes with conditions (0.37 vs 0.30: glasses, camera angle, lighting), so per-user calibration is needed.
4. People blink naturally without noticing, so natural blinks must be ignored.
5. Time resolution is limited by the camera (~30 fps → steps of ~0.033 s).