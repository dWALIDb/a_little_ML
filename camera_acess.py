# mediapipe is not compatible with python 3.13 so we gotta run it with python 3.10
# this  means we run with the flag  py -3.10 and it comes with all the tools too :)

import cv2

# access the cam device (default)
cap=cv2.VideoCapture(0)
# check status of opened device
if not cap.isOpened():
    print("camera is not accessed :(")
    exit(1)

# standard open cv boiler plate 
while True:
    # read the camera frame
    ret,frame = cap.read()
    if not ret:
        print("ENDED FRAMES")
        break

    
    # create a window that shows the camera frame :)
    cv2.imshow("frame", frame)
    
    
    
    # cv2.putText(frame,"HELLO WORLD",(100,100),cv2.FONT_HERSHEY_DUPLEX,4,(255,255,0),2,cv2.LINE_AA)
    if cv2.waitKey(2)==ord('q'): 
        break
    # this destroys the window when x window button s pressed  
    if cv2.getWindowProperty('frame',cv2.WND_PROP_VISIBLE) < 1:        
        break 


cap.release()
cv2.destroyAllWindows()