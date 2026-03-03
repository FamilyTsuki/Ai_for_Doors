import ctypes
import numpy as np
import cv2
import mss

# 1. Charger la bibliothèque C++
slam_lib = ctypes.CDLL('./slam_lib.dll')

with mss.mss() as sct:
    monitor = {"top": 0, "left": 0, "width": 640, "height": 640}
    
    while True:
        # Capture l'écran
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 2. Envoyer l'image au C++ pour le SLAM
        # On passe le pointeur de données pour que ce soit ultra rapide
        ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        slam_lib.process_frame(640, 640, ptr)

        # 3. Ici, ton YOLO continue de tourner en parallèle
        # results = model.predict(frame)
        
        cv2.imshow("SLAM + YOLO Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break