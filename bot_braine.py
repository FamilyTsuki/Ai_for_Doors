import ctypes
import cv2
import numpy as np
import mss
import os
import time
import pyautogui
from ultralytics import YOLO

# --- CONFIGURATION ---
# Charge le modèle YOLOv8 Nano (léger pour les jeux)
model = YOLO('yolov11n.pt') 

dll_path = os.path.abspath("./slam_lib.dll")
slam_lib = ctypes.CDLL(dll_path, winmode=0)
slam_lib.process_frame.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_int)]
slam_lib.process_frame.restype = ctypes.c_float

# --- VARIABLES DU CERVEAU ---
nb_points = ctypes.c_int(0)
historique_mouvement = []
dernier_mouvement_clavier = ""

def appuyer_touche(touche, duree=0.1):
    """Simule l'appui d'une touche du clavier"""
    pyautogui.keyDown(touche)
    time.sleep(duree)
    pyautogui.keyUp(touche)

with mss.mss() as sct:
    monitor = {"top": 100, "left": 100, "width": 640, "height": 640}
    print("🧠 Cerveau en marche. Switch sur Doors !")
    time.sleep(3) # Laisse le temps de cliquer sur le jeu

    while True:
        # 1. Perception (Capture + SLAM + YOLO)
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # SLAM (Mouvement)
        data_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        mouv_brut = slam_lib.process_frame(640, 640, data_ptr, ctypes.byref(nb_points))
        
        # Lissage
        historique_mouvement.append(mouv_brut)
        if len(historique_mouvement) > 5: historique_mouvement.pop(0)
        mouv_lisse = sum(historique_mouvement) / len(historique_mouvement)

        # YOLO (Objets)
        results = model(frame, conf=0.5, verbose=False)
        detections = results[0].boxes

        # 2. Analyse (Le Cerveau décide)
        action = "AVANCER"
        confiance_action = (0, 255, 0)

        # RÈGLE 1 : Si on est bloqué contre un mur (SLAM)
        if mouv_lisse < 0.1 and nb_points.value > 20:
            action = "EVITEMENT (MUR)"
            confiance_action = (0, 0, 255)
            # Manoeuvre : Reculer et décaler
            appuyer_touche('s', 0.2)
            appuyer_touche('d', 0.3) 

        # RÈGLE 2 : Si YOLO voit une porte (Cible)
        cible_detectee = False
        for box in detections:
            # On cherche par exemple une porte (classe 0 dans un modèle Doors personnalisé)
            # Ici on utilise le modèle de base pour le test
            if model.names[int(box.cls[0])] in ['door', 'refrigerator']: 
                cible_detectee = True
                x1, y1, x2, y2 = box.xyxy[0]
                centre_x = (x1 + x2) / 2
                
                # S'aligner avec la cible
                if centre_x < 280: # Trop à gauche
                    appuyer_touche('q', 0.1)
                elif centre_x > 360: # Trop à droite
                    appuyer_touche('d', 0.1)
                break

        # RÈGLE 3 : Si tout va bien, on avance
        if action == "AVANCER":
            appuyer_touche('z', 0.1)

        # 3. Affichage (Debug)
        cv2.rectangle(frame, (10, 10), (350, 60), (0,0,0), -1)
        cv2.putText(frame, f"ACTION: {action}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, confiance_action, 2)
        
        # Dessine les boîtes YOLO
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        cv2.imshow("IA Cerveau - Doors Bot", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()