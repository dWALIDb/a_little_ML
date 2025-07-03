import mediapipe as mp
import os
import cv2
import csv
import math

distances = []

def euclidean_distance(a, b)->float:
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

BASE_PATH=r"C:\Users\DELL\Desktop\learn\python\dataset"

categories= ["thumbs_up","thumbs_down","non_wanted"]

thumbs_up= os.listdir(os.path.join(BASE_PATH,"thumbs_up"))
thumbs_down= os.listdir(os.path.join(BASE_PATH,"thumbs_down"))
non_wanted= os.listdir(os.path.join(BASE_PATH,"non_wanted"))

lists_concated=[thumbs_up,thumbs_down,non_wanted]
print(lists_concated)

#  getting drawing utils and hands modules to modify images later
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Hands constructor gives some parameters for detection and tracking confidences
hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,max_num_hands=5)

# set up csv to write landmarks [landmark_index , x , y , z] these are relative to wrist position
# because we need info about the gestures and not the place of it in image


for elem,it  in zip(lists_concated,categories):
    path=os.path.join(BASE_PATH,it)
    csv_file = open(os.path.join(BASE_PATH,"csv_"+ it +"_landmarks.csv"),mode="w",newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['landmark index', "distance to wrist"])

    for i in range(len(elem)):
        print(os.path.join(path,elem[i]))
        image = cv2.imread(os.path.join(path,elem[i]),cv2.IMREAD_UNCHANGED)
        if image is None: 
            print("image not found")
            continue
        
        # Ensure the image has 3 channels 3 dimensions or 3 channels 
        if len(image.shape) != 3 or image.shape[2] < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] > 3: 
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        image_rgb= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        # print(image.shape)
        if result.multi_hand_landmarks is None: 
            print(f"no featues") 
            continue
        
        
        # calculated distances according to wrist distance and hand size
        for landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image,landmarks,mp_hands.HAND_CONNECTIONS)
            for i in range(1,len(landmarks.landmark)):
                d=euclidean_distance(landmarks.landmark[i],
                            landmarks.landmark[mp_hands.HandLandmark.WRIST])
                writer.writerow([i,d])