import cv2

cap = cv2.VideoCapture(1)          # camera 1 = the Mac's built-in camera

if not cap.isOpened():             # NEW: check the camera opened at all
    print("Camera could not be opened")
    exit()

fails = 0                          # NEW: counts empty frames in a row

while True:
    ok, frame = cap.read()         # grab one image from the camera
    if not ok:
        fails = fails + 1          # NEW
        if fails > 100:            # NEW: give up only after 100 empty frames
            print("Camera never started")
            break
        continue                   # NEW: skip this round, try the next frame
    fails = 0                      # NEW: got a picture, reset the counter

    frame = cv2.flip(frame, 1)     # mirror it
    cv2.putText(frame, "Press q to quit", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("BlinkTalk", frame) # show the image in a window

    if cv2.waitKey(1) & 0xFF == ord("q"):  # quit when you press q
        break

cap.release()
cv2.destroyAllWindows()