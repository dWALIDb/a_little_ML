import cv2
import mediapipe as mp

# changes normalized landmark values to pixel coordinates
def denormalize(point ,width,height)->list[int]:
 return [int(point.x * width),int(point.y * height), int(point.z * 100)]
    

# Initialize MediaPipe Hands.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Hands constructor gives some parameters for detection and tracking confidences
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Open webcam. the default one
cap = cv2.VideoCapture(0)


while cap.isOpened():
    #read image into buffer and to use 
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display.
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands.
    results = hands.process(image_rgb)
    # Draw hand landmarks.
    if results.multi_hand_landmarks:
        # use loop when detecting more than one hand
        # for hand_landmarks in results.multi_hand_landmarks:
        hand_landmarks= results.multi_hand_landmarks[0]

        thumb_pos= hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_pos= hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp= hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        thumb_pos_d=denormalize(thumb_pos,image.shape[1],image.shape[0])
        index_pos_d=denormalize(index_pos,image.shape[1],image.shape[0])
        index_mcp_d=denormalize(index_mcp,image.shape[1],image.shape[0])

        if index_mcp_d[1]>=thumb_pos_d[1]:
            cv2.putText(image,f" - UP {thumb_pos_d[1]=} - ",(50,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2,cv2.LINE_AA)
        else:
            cv2.putText(image,f" - DOWN {thumb_pos_d[1]=} - ",(50,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2,cv2.LINE_AA)

        
        # z axis is negative when pointing to camera 
        # positive when pointing away
        #draw 2 lines that start from the index and pinkey tips
        # cv2.line(image,(index_pos_d[0],index_pos_d[1]),(thumb_pos_d[0],thumb_pos_d[1]),(255,0,0),2,cv2.LINE_AA)
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Show the image.
    cv2.imshow('MediaPipe Hands', image)

    if (cv2.waitKey(1) & 0xFF) == 27:  # Press 'Esc' to exit
        break
    if (cv2.getWindowProperty('MediaPipe Hands', cv2.WND_PROP_VISIBLE))<1:
        break


cap.release()
cv2.destroyAllWindows()
