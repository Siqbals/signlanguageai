#imports 
import mediapipe as mp 
import cv2 
import os
import pickle


#hand detection vars, as well as tools to detect landmarks in the hands
#this is from the mediapipe library 
mphands = mp.solutions.hands
mpdrawings = mp.solutions.drawing_utils
mpdrawingstyles = mp.solutions.drawing_styles 

#static image mode - process each image standalone and independent 
#confidence - sensitivity of detecting a hand 
hands = mphands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#where our camera data is located 
DATA_DIR = './data'

#contain the data info
data = []

#catagories for each img 
labels = []

#itereate thru each directory (letter) in the dataset
for dir in os.listdir(DATA_DIR):


    #now iterate thru each image in 'data/<dir name>'
    for imgpath in os.listdir(os.path.join(DATA_DIR, dir)):
        #save x and y coordinates here 
        dataaux = []
    
        #read each image and convert rgb
        img = cv2.imread(os.path.join(DATA_DIR, dir, imgpath))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #detect all landmarks in all images
        results = hands.process(img_rgb)

        #grab x and y values for each images landmark and store 
        for landmarks in results.multi_hand_landmarks:
            for i in range(len(landmarks.landmark)):
                xcord = landmarks.landmark[i].x
                ycord = landmarks.landmark[i].y
                dataaux.append(xcord)
                dataaux.append(ycord)
                
        data.append(dataaux)
        labels.append(dir)

#get the data and store it in a file, creating out data set
f = open('landmark.pickle', 'wb')
pickle.dump({'data':data, 'labels': labels}, f)
f.close()