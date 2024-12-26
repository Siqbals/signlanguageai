import cv2 
import mediapipe as mp
import pickle
import numpy as np


modeldict = pickle.load(open('./model.p', 'rb'))
model = modeldict['model']
cam = cv2.VideoCapture(0)

mphands = mp.solutions.hands
mpdrawings = mp.solutions.drawing_utils
mpdrawingstyles = mp.solutions.drawing_styles

hands = mphands.Hands(static_image_mode=True, min_detection_confidence=0.3) 

labelsdict = {0: 'A', 1: 'B', 2: 'C' , 3: 'D', 4: 'E',
               5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
               15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 
               25: 'Z' 
            }

while True:

    dataaux = []
    xarr = []
    yarr = []

    

    ret, frame = cam.read()
    H,W, _ = frame.shape

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(framergb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mpdrawings.draw_landmarks(
                frame,
                landmarks,
                mphands.HAND_CONNECTIONS,
                mpdrawingstyles.get_default_hand_landmarks_style(),
                mpdrawingstyles.get_default_hand_connections_style()
            )
        
        for i in range(len(landmarks.landmark)):
            x = landmarks.landmark[i].x
            y = landmarks.landmark[i].y

            xarr.append(x)
            yarr.append(y)
            
        for landmarks in results.multi_hand_landmarks:
            for i in range(len(landmarks.landmark)):
                xcord = landmarks.landmark[i].x
                ycord = landmarks.landmark[i].y
                dataaux.append(xcord)
                dataaux.append(ycord)

        x1 = int(min(xarr) * W)
        y1 = int(min(yarr) * H)
        
        x2 = int(max(xarr) * W)
        y2 = int(max(yarr) * H)
        
        prediction = model.predict([np.asarray(dataaux)])
        predictedchar = labelsdict[int(prediction[0])]
        
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 4)
        cv2.putText(frame, predictedchar, (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,0,0), 3, cv2.LINE_AA)


        

    cv2.imshow('frame', frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()

'''

'''