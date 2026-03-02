import cv2
import numpy as np
import mss
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

with mss.mss() as sct:
    monitor = {"top": 10, "left": 10, "width": 940, "height": 940}

    print("Détection en cours... Appuie sur 'q' pour quitter.")

    while True:
        img = np.array(sct.grab(monitor))
        
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = model.predict(frame, conf=0.8, imgsz=320, verbose=False)

        annotated_frame = results[0].plot()

        cv2.imshow("IA Doors - Vision", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()