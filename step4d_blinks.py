# ===================== IMPORTS: bring in the tools we need =====================
import time                                       # time: lets us read the clock (for timestamps)
import cv2                                        # OpenCV: camera, drawing, showing windows
import numpy as np                                #NumPy, math tools (distances), nicknamed "np"
import mediapipe as mp                            # MediaPipe, nicknamed "mp" so we type less
from mediapipe.tasks import python as mp_python   # MediaPipe's general "tasks" settings (BaseOptions)
from mediapipe.tasks.python import vision         # MediaPipe's image tools, including the face finder

# ===================== EYE POINTS =====================
# The 6 landmark numbers for each eye, in the order p1..p6 from the paper:
# p1 & p4 = eye corners, p2 & p3 = upper lid, p6 & p5 = lower lid
RIGHT_EYE = [33, 160, 158, 133, 153, 144]         # landmark numbers for the right eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]         # landmark numbers for the left eye

# ===================== MY PERSONAL EAR VALUES (measured in step 4b) =====================
OPEN_EAR = 0.35                                   # my EAR with eyes normally open
CLOSED_EAR = 0.03                                 # my EAR with eyes closed
EAR_THRESHOLD = (OPEN_EAR + CLOSED_EAR) / 2       # halfway between them = 0.275
MIN_BLINK = 0.05                                  # ignore "blinks" shorter than 0.05 s (noise, 1 jittery frame)

# ===================== FUNCTIONS =====================
def get_eye_points(face, eye, w, h):              # define a function with 4 inputs
    """Turn MediaPipe's 6 eye landmarks into 6 (x, y) pixel positions."""  # description of the function
    points = []                                   # start with an empty list
    for i in eye:                                 # go through the 6 landmark numbers of this eye
        x = face[i].x * w                         # fraction × width → pixel x
        y = face[i].y * h                         # fraction × height → pixel y
        points.append([x, y])                     # add this point [x, y] to the end of the list
    return np.array(points)                       # give back the 6 points as a NumPy array (so we can do math)


def eye_aspect_ratio(pts):                        # define the EAR function; input = 6 points
    """EAR formula from Soukupova & Cech (2016)."""  # description of the function
    p1, p2, p3, p4, p5, p6 = pts                  # give each of the 6 points its name from the paper
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)  # the two eye heights added together
    horizontal = 2.0 * np.linalg.norm(p1 - p4)    # 2 × the eye width (corner to corner)
    return vertical / horizontal                  # EAR = heights / (2 × width); give the answer back

def draw_text(frame, text, pos, scale=0.8, color=(255, 255, 255)):
    """Draw text with a black outline so it's readable on any background."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 5, cv2.LINE_AA)        # 1st: thick black text = the outline
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)            # 2nd: thinner coloured text on top

# ===================== SETTING UP THE FACE FINDER (done once) =====================
options = vision.FaceLandmarkerOptions(           # start filling in the "settings form"
    base_options=mp_python.BaseOptions(model_asset_path="models/face_landmarker.task"),  # where the trained AI model file is
    running_mode=vision.RunningMode.VIDEO,        # we give it a video stream, so it can track the face between frames
    num_faces=1,                                  # look for only one face (faster, no confusion)
)                                                 # end of the settings form
landmarker = vision.FaceLandmarker.create_from_options(options)  # build the face finder from those settings

# ===================== OPENING THE CAMERA =====================
cap = cv2.VideoCapture(1)                         # connect to camera 1 (the Mac's built-in camera; 0 was the iPhone)
if not cap.isOpened():                            # if the camera could not be connected...
    print("Camera could not be opened")           # ...tell the user
    exit()                                        # ...and stop the program

fails = 0                                         # counter: how many empty frames in a row
start = time.time()                               # remember the time when the program started (in seconds)

# ===================== MEMORY THAT LASTS BETWEEN FRAMES =====================
eye_was_closed = False                            # was the eye closed in the previous frame?
closed_start = 0.0                                # the time when the eye last closed (stopwatch start)
blink_count = 0                                   # how many blinks so far
last_duration = 0.0                               # how long the last blink lasted (seconds)

# ===================== MAIN LOOP: runs once per frame, ~30 times per second =====================
while True:                                       # repeat forever (until we "break")
    ok, frame = cap.read()                        # take one picture; ok = did it work?, frame = the picture
    if not ok:                                    # if no picture came back (camera still warming up)...
        fails = fails + 1                         # ⚠️ FIXED: count this failure
        if fails > 100:                           # ⚠️ FIXED: if 100 failures in a row...
            print("Camera never started")         # ⚠️ FIXED: ...tell the user
            break                                 # ⚠️ FIXED: ...and leave the loop
        continue                                  # skip the rest of this round, try the next frame
    fails = 0                                     # we got a picture, so reset the failure counter
    frame = cv2.flip(frame, 1)                    # mirror the picture left↔right, like a mirror

    # ----- Colour conversion and MediaPipe -----
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV uses Blue-Green-Red; MediaPipe wants Red-Green-Blue
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # wrap the picture in MediaPipe's own format
    timestamp_ms = int((time.time() - start) * 1000)  # milliseconds since the program started
    result = landmarker.detect_for_video(mp_image, timestamp_ms)  # the AI step: find the face and its ~478 points

    if result.face_landmarks:                     # if at least one face was found...
        face = result.face_landmarks[0]           # take the first face
        h, w = frame.shape[:2]                    # picture height and width in pixels

        # ----- Computing EAR -----
        right_pts = get_eye_points(face, RIGHT_EYE, w, h)  # 6 pixel points of the right eye
        left_pts = get_eye_points(face, LEFT_EYE, w, h)    # 6 pixel points of the left eye
        right_ear = eye_aspect_ratio(right_pts)            # EAR of the right eye
        left_ear = eye_aspect_ratio(left_pts)              # EAR of the left eye
        ear = (right_ear + left_ear) / 2                   # average of both eyes

        # ----- OPEN or CLOSED? -----
        if ear < EAR_THRESHOLD:                   # below the line → eyes closed
            eye_state = "CLOSED"                  # remember the state as text
            color = (0, 0, 255)                   # red (BGR)
        else:                                     # above the line → eyes open
            eye_state = "OPEN"                    # ★ NEW
            color = (0, 255, 0)                   # green (BGR)
        
        # ----- Drawing eye points (now in the state's colour) -----
        for i in RIGHT_EYE + LEFT_EYE:            # go through all 12 eye landmark numbers
            x = int(face[i].x * w)                # fraction × width → pixel x
            y = int(face[i].y * h)                # fraction × height → pixel y
            cv2.circle(frame, (x, y), 3, color, -1)  # ★ CHANGED: dot colour = green if open, red if closed
        
        # ----- Detecting blinks (state changes) -----
        now = time.time()                         # current time in seconds

        if eye_state == "CLOSED" and not eye_was_closed:   # it's closed now, but was open before → it JUST closed
            closed_start = now                    # start the stopwatch
            eye_was_closed = True                 # remember it's closed

        elif eye_state == "OPEN" and eye_was_closed:       # it's open now, but was closed before → it JUST opened
            duration = now - closed_start         # stop the stopwatch: how long was it closed?
            eye_was_closed = False                # remember it's open
            if duration >= MIN_BLINK:             # long enough to be a real blink (not noise)?
                blink_count = blink_count + 1     # count it
                last_duration = duration          # remember its duration for the screen
                print(f"Blink {blink_count}: {duration:.2f} s")  # also print it in Terminal (a record!)
        
       
        # ----- Showing values on screen (with outlines) -----
        draw_text(frame, f"EAR: {ear:.2f}", (20, 40), 1)                        # EAR value, white
        draw_text(frame, f"Eye: {eye_state}", (20, 80), 1, color)               # OPEN/CLOSED in green/red
        draw_text(frame, f"Threshold: {EAR_THRESHOLD:.3f}", (20, 115), 0.6)     # threshold, smaller
        draw_text(frame, f"Blink count: {blink_count}", (20, 150))              # number of blinks
        draw_text(frame, f"Last blink duration: {last_duration:.2f} s", (20, 185))  # last duration
        if eye_was_closed:                                                      # while the eye is closed...
            draw_text(frame, f"Closed for: {now - closed_start:.2f} s", (20, 220), 0.8, (0, 0, 255))  # ...red stopwatch

    else:                                         # if no face was found...
        draw_text(frame, "No face", (20, 40), 1, (0, 0, 255))   # "No face" in red, with outline
    
    # ----- Show, quit -----
    cv2.imshow("BlinkTalk", frame)                # show the picture (with the dots) in a window called "BlinkTalk"
    if cv2.waitKey(1) & 0xFF == ord("q"):         # wait 1 ms for a key; if it was "q"...
        break                                     # ...leave the loop

# ===================== CLEANUP (after the loop ends) =====================
cap.release()                                     # switch the camera off and let other apps use it
landmarker.close()                                # switch the AI model off and free its memory
cv2.destroyAllWindows()                           # close the window