import mediapipe as mp
import cv2 

PATH=r"C:\Users\DELL\Desktop\learn\python\dataset\thumbs_up8.jpg"
image = cv2.imread(PATH,cv2.IMREAD_COLOR)
if  image is None:
    print("image not found") 
    exit()
resized = cv2.resize(image,(800,600))
image_rgb = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Hands constructor gives some parameters for detection and tracking confidences
hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.8, min_tracking_confidence=0.8)

result = hands.process(image_rgb)

if not result.multi_hand_landmarks:
    print("couldn't find hands") 
    exit()

mp_drawing.draw_landmarks(resized, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

cv2.imshow("image",resized)
cv2.setWindowProperty("image",cv2.WND_PROP_AUTOSIZE,1)
cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,1)
cv2.waitKey(0)
cv2.destroyAllWindows()