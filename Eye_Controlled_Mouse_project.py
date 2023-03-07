import cv2
import mediapipe as mp
import pyautogui
import numpy as np

#####################
wCam, hCam = 640, 480
frameR = 100 #frame reduction
smoothening = 10
#####################

plocX, plocY = 0, 0
clocX, clocY = 0, 0


cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
# screen_w, screen_h = autopy.screen.size()

while True:
    _, frame = cam.read()
    # frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    # get the dimension of frame
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))          
            if id == 1:
                 #5 convert cordinates          
                x3 = np.interp(x,(frameR, wCam-frameR), (0,screen_w))
                y3 = np.interp(y,(frameR, hCam-frameR), (0, screen_h))

                #6 smoothen values
                clocX = plocX  + (x3 - plocX) / smoothening
                clocY = plocX  + (y3 - plocY) / smoothening

                # screen_x = screen_w * landmark.x
                # screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_w-clocX, clocY)
                plocX, plocY = clocX, clocY
        left = [landmarks[145], landmarks[159]]
        
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 5, (0, 255, 255))
        if (left[0].y - left[1].y) < 0.005:
            pyautogui.click()
            pyautogui.sleep(1)
    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)





