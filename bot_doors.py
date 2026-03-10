import ctypes
import cv2
import numpy as np
import mss
import os
import time
import pyautogui
from ultralytics import YOLO

# --- CHARGEMENT ---
model = YOLO('yolov8n.pt') 
slam_lib = ctypes.CDLL(os.path.abspath("./slam_lib.dll"), winmode=0)
slam_lib.process_frame.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_int)]
slam_lib.process_frame.restype = ctypes.c_float

# --- CONFIG DU CERVEAU ---
LARGEUR_ECRAN = 640
CENTRE_X = LARGEUR_ECRAN // 2
ZONE_MORTE = 50 # Marge d'erreur au centre (pixels)
nb_points = ctypes.c_int(0)
historique_mouv = []

def action_clavier(touche, duree=0.05):
    pyautogui.keyDown(touche)
    time.sleep(duree)
    pyautogui.keyUp(touche)

print("🚀 Lancement du Chasseur de Portes...")
time.sleep(3) # Temps pour cliquer sur la fenêtre Doors

with mss.mss() as sct:
    monitor = {"top": 100, "left": 100, "width": 640, "height": 640}

    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        data_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        
        # 1. Analyse du mouvement (SLAM)
        mouv = slam_lib.process_frame(640, 640, data_ptr, ctypes.byref(nb_points))
        historique_mouv.append(mouv)
        if len(historique_mouv) > 5: historique_mouv.pop(0)
        mouv_moyen = sum(historique_mouv) / len(historique_mouv)

        # 2. Vision YOLO
        results = model(frame, conf=0.3, verbose=False) # Confiance basse pour les tests
        detections = results[0].boxes
        
        cible_x = None
        for box in detections:
            # On cherche des objets rectangulaires (portes, cadres)
            # Dans le modèle de base, on peut tester 'door' (classe 0 ou 1)
            cls = int(box.cls[0])
            label = model.names[cls]
            
            # Pour Doors, on cible souvent ce que YOLO prend pour un meuble ou une porte
            if label in ['door', 'refrigerator', 'potted plant', 'tv']: 
                x1, y1, x2, y2 = box.xyxy[0]
                cible_x = (x1 + x2) / 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                break

        # 3. Logique de Navigation
        etat = "RECHERCHE"
        
        if mouv_moyen < 0.1 and nb_points.value > 20:
            # ANTI-BLOCAGE : On fonce dans un mur
            etat = "EVITEMENT MUR"
            action_clavier('s', 0.2)
            action_clavier('d', 0.3)
        
        elif cible_x is not None:
            # GUIDAGE VERS LA PORTE
            delta = cible_x - CENTRE_X
            
            if delta < -ZONE_MORTE:
                etat = "CORRECTION GAUCHE"
                action_clavier('q', 0.05)
            elif delta > ZONE_MORTE:
                etat = "CORRECTION DROITE"
                action_clavier('d', 0.05)
            else:
                etat = "DROIT DEVANT"
                action_clavier('z', 0.1)
        else:
            # Si aucune porte vue, on avance prudemment ou on tourne la caméra
            etat = "AVANCE PRUDENTE"
            action_clavier('z', 0.05)

        # DEBUG VISUEL
        cv2.putText(frame, f"ETAT: {etat}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Cerveau de Navigation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cv2.destroyAllWindows()