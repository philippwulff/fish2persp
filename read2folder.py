import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

for i in range(50):
    ret, frame = cap.read()
    
    cv2.imwrite(f"calib_files/{i}.jpg", frame)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()


