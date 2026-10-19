# ===================== IMPORTS: bring in the tools we need =====================
import time                                       # time: lets us read the clock (for timestamps)
import cv2                                        # OpenCV: camera, drawing, showing windows
import mediapipe as mp                            # MediaPipe, nicknamed "mp" so we type less
from mediapipe.tasks import python as mp_python   # MediaPipe's general "tasks" settings (BaseOptions)
from mediapipe.tasks.python import vision         # MediaPipe's image tools, including the face finder

# ===================== EYE POINTS =====================
# The 6 landmark numbers for each eye, in the order p1..p6 from the paper:
# p1 & p4 = eye corners, p2 & p3 = upper lid, p6 & p5 = lower lid
RIGHT_EYE = [33, 160, 158, 133, 153, 144]         # landmark numbers for the right eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]         # landmark numbers for the left eye

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

    # ----- Drawing eye points -----
    if result.face_landmarks:                     # if at least one face was found...
        face = result.face_landmarks[0]           # take the first face (Python counts from 0)
        h, w = frame.shape[:2]                    # get the picture's height and width in pixels
        for i in RIGHT_EYE + LEFT_EYE:            # go through all 12 eye landmark numbers, one at a time
            x = int(face[i].x * w)                # x is a fraction (0–1); × width → pixel column
            y = int(face[i].y * h)                # y is a fraction (0–1); × height → pixel row
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # draw a filled green dot (radius 3) at that point
    else:                                         # if no face was found...
        cv2.putText(frame, "No face", (20, 40),   # write "No face" near the top-left corner...
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # ...font, size 1, red (BGR), thickness 2

    # ----- Show, quit -----
    cv2.imshow("BlinkTalk", frame)                # show the picture (with the dots) in a window called "BlinkTalk"
    if cv2.waitKey(1) & 0xFF == ord("q"):         # wait 1 ms for a key; if it was "q"...
        break                                     # ...leave the loop

# ===================== CLEANUP (after the loop ends) =====================
cap.release()                                     # switch the camera off and let other apps use it
landmarker.close()                                # switch the AI model off and free its memory
cv2.destroyAllWindows()                           # close the window