import ctypes
import cv2
import numpy as np
import mss
import os
import time
import pydirectinput
from ultralytics import YOLO

# --- CONFIGURATION ---
PATH_MODEL = "runs/detect/train/weights/best.pt"
model = YOLO(PATH_MODEL)

# Paramètres de fluidité
SENSIVITE_CAMERA = 0.15   
SMOOTHING = 0.2           
ZONE_MORTE = 45           
OFFSET_X = 0              
pydirectinput.PAUSE = 0

# --- MOTEUR DE SOURIS MATÉRIEL (WIN32) ---
MOUSEEVENTF_MOVE = 0x0001  # Constante Windows pour le mouvement relatif

def move_mouse_hardware(dx):
    """Injecte le mouvement directement dans le flux d'entrée Windows via mouse_event"""
    if abs(dx) > 0.1:
        # On force un minimum de 2 pixels pour "réveiller" le Raw Input de Roblox
        val_x = int(dx)
        if 0 < dx < 2: val_x = 2
        elif -2 < dx < 0: val_x = -2
        
        # Appel direct à l'API Windows (Nécessite VS Code en mode Administrateur)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, val_x, 0, 0, 0)

# Chargement DLL SLAM
dll_path = os.path.abspath("./slam_lib.dll")
slam_lib = ctypes.CDLL(dll_path, winmode=0)
slam_lib.process_frame.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_int)]
slam_lib.process_frame.restype = ctypes.c_float

LARGEUR, HAUTEUR = 1200, 940
CENTRE_X_REEL = (LARGEUR // 2) + OFFSET_X

nb_points = ctypes.c_int(0)
historique_mouv = []

print("🚀 IA Doors - Moteur Win32 Hardware Actif")
print("⚠️ OBLIGATOIRE : Lance VS Code en tant qu'Administrateur !")
time.sleep(3)

with mss.mss() as sct:
    monitor = {"top": 10, "left": 10, "width": LARGEUR, "height": HAUTEUR}

    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        data_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        
        # 1. SLAM
        mouv = slam_lib.process_frame(LARGEUR, HAUTEUR, data_ptr, ctypes.byref(nb_points))
        historique_mouv.append(mouv)
        if len(historique_mouv) > 10: historique_mouv.pop(0)
        mouv_moyen = sum(historique_mouv) / len(historique_mouv)

        # 2. Vision YOLO
        results = model.predict(frame, conf=0.5, imgsz=320, verbose=False)
        detections = results[0].boxes
        
        cible_x = None
        if len(detections) > 0:
            box = sorted(detections, key=lambda b: abs(((b.xyxy[0][0]+b.xyxy[0][2])/2) - CENTRE_X_REEL))[0]
            x1, y1, x2, y2 = box.xyxy[0]
            cible_x = (x1 + x2) / 2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.line(frame, (int(cible_x), 0), (int(cible_x), HAUTEUR), (255, 0, 0), 1)

        # 3. Logique de Navigation
        status = "EXPLORATION"
        
        if cible_x is not None:
            erreur = cible_x - CENTRE_X_REEL
            
            # Rotation via mouse_event
            if abs(erreur) > ZONE_MORTE:
                mouv_final = (erreur * SENSIVITE_CAMERA) * SMOOTHING
                move_mouse_hardware(mouv_final)
                status = "CIBLAGE"
            
            # Marche
            if abs(erreur) < 200:
                pydirectinput.keyDown('w')
                time.sleep(0.08)
                pydirectinput.keyUp('w')
                status = "MARCHE"

            # Sécurité SLAM
            if mouv_moyen < 0.02 and nb_points.value > 15:
                status = "BLOCAGE !"
                move_mouse_hardware(150) # Grand coup de tête pour se dégager
                pydirectinput.keyDown('s')
                time.sleep(0.4)
                pydirectinput.keyUp('s')
                historique_mouv.clear()

        else:
            # SCAN (Recherche)
            status = "SCANNING..."
            move_mouse_hardware(3) # Rotation lente via mouse_event
            if time.time() % 3 > 2.5:
                pydirectinput.press('w')

        # Debug
        cv2.circle(frame, (CENTRE_X_REEL, HAUTEUR//2), 6, (0,0,255), -1)
        cv2.putText(frame, f"ETAT: {status}", (20, 50), 1, 1.5, (0, 255, 0), 2)
        cv2.imshow("IA Doors - Win32 Hardware", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()
