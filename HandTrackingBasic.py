import cv2
import mediapipe as mp
import time

# Camera Capture
cap = cv2.VideoCapture(0)

# Hands attribute
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Draw attribute
mpDraw = mp.solutions.drawing_utils

# FPS attribute
pTime = 0
cTime = 0

while(True):
    # Capture frame-by-frame
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Checking the hands detected or not
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # looking for the locations with coordinates
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                width, height, depth = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                print(f"ID {id} : x={cx}, y={cy}")

                # Identifying the fingers tip (MASIH NGEBUG ATO LAPTOP AKU YANG KENTANG......)               
                #if id == 4 : # Thumb tip 
                    #cv2.circle(img, (cx, cy), 7, (0, 255, 255), cv2.FILLED) # (video, location, radius, fill)
                
            # Draw the hands
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Write the FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS {str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (170, 255, 0), 4) # (video, text, location on video, font, size, color, thickness)

    # Display the resulting frame
    cv2.imshow('image',img)
    cv2.waitKey(1)