import ctypes
import cv2
import numpy as np
import mss
import os
import time
import pydirectinput
from ultralytics import YOLO

# ==========================================
# CONFIGURATION GÉNÉRALE
# ==========================================
PATH_MODEL = "runs/detect/train/weights/best.pt"
model = YOLO(PATH_MODEL)

# Paramètres de mouvement (Optimisés pour Roblox)
SENSIVITE_CAMERA = 0.12   
SMOOTHING = 0.18          
ZONE_MORTE = 50           
VITESSE_SCAN = 4
pydirectinput.PAUSE = 0

# Dimensions de capture (Doit correspondre à ton entraînement)
LARGEUR, HAUTEUR = 1200, 940
CENTRE_X = LARGEUR // 2

# ==========================================
# MOTEUR BAS NIVEAU (SOURIS & CLAVIER)
# ==========================================
MOUSEEVENTF_MOVE = 0x0001

def move_mouse_hardware(dx):
    """Contourne le Raw Input de Roblox via l'API Win32 brute"""
    if abs(dx) > 0.5:
        # Force un minimum de pixels pour réveiller le moteur physique
        val_x = int(dx) if abs(dx) >= 2 else (2 if dx > 0 else -2)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, val_x, 0, 0, 0)

# ==========================================
# PERCEPTION SLAM
# ==========================================
dll_path = os.path.abspath("./slam_lib.dll")
slam_lib = ctypes.CDLL(dll_path, winmode=0)
slam_lib.process_frame.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_int)]
slam_lib.process_frame.restype = ctypes.c_float

# ==========================================
# BOUCLE PRINCIPALE D'INTELLIGENCE
# ==========================================
nb_points = ctypes.c_int(0)
historique_mouv = []
derniere_porte_vue = 0
direction_exploration = 1

print("🚀 INITIALISATION DE L'IA DOORS...")
print("⚠️ MODE ADMINISTRATEUR REQUIS")
time.sleep(3)

with mss.mss() as sct:
    monitor = {"top": 10, "left": 10, "width": LARGEUR, "height": HAUTEUR}

    while True:
        # 1. CAPTURE & PERCEPTION
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        data_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        
        # Calcul du mouvement réel via SLAM
        mouv = slam_lib.process_frame(LARGEUR, HAUTEUR, data_ptr, ctypes.byref(nb_points))
        historique_mouv.append(mouv)
        if len(historique_mouv) > 15: historique_mouv.pop(0)
        mouv_moyen = sum(historique_mouv) / len(historique_mouv)

        # 2. VISION (YOLO)
        results = model.predict(frame, conf=0.55, imgsz=320, verbose=False)
        detections = results[0].boxes
        
        cible_x = None
        if len(detections) > 0:
            # On cible la porte la plus proche de notre axe central
            box = sorted(detections, key=lambda b: abs(((b.xyxy[0][0]+b.xyxy[0][2])/2) - CENTRE_X))[0]
            x1, y1, x2, y2 = box.xyxy[0]
            cible_x = (x1 + x2) / 2
            derniere_porte_vue = time.time()
            
            # Dessin Debug
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.line(frame, (int(cible_x), 0), (int(cible_x), HAUTEUR), (255, 0, 0), 2)

        # 3. PRISE DE DÉCISION (LE CERVEAU)
        status = "EN ATTENTE"
        color = (255, 255, 255)

        # SCÉNARIO A : ON VOIT UNE PORTE (Navigation active)
        if cible_x is not None:
            erreur = cible_x - CENTRE_X
            # Rotation fluide
            if abs(erreur) > ZONE_MORTE:
                move_mouse_hardware((erreur * SENSIVITE_CAMERA) * SMOOTHING)
            
            # Marche vers l'objectif
            pydirectinput.keyDown('w')
            status = "NAVIGATION VERS PORTE"
            color = (0, 255, 0)

        # SCÉNARIO B : ON EST BLOQUÉ (SLAM)
        # Si on est censé avancer mais que le décor ne bouge plus
        elif mouv_moyen < 0.015 and nb_points.value > 15:
            status = "BLOCAGE DÉTECTÉ - RECALCUL"
            color = (0, 0, 255)
            pydirectinput.keyUp('w')
            pydirectinput.keyDown('s') # Recul
            time.sleep(0.5)
            pydirectinput.keyUp('s')
            move_mouse_hardware(250 * direction_exploration) # Tourne la tête
            historique_mouv.clear()

        # SCÉNARIO C : TRANSITION (On vient de passer une porte)
        elif time.time() - derniere_porte_vue < 1.5:
            status = "ENTRÉE DANS LA SALLE"
            pydirectinput.keyDown('w') # On continue de marcher pour s'éloigner du mur
            color = (255, 250, 0)

        # SCÉNARIO D : EXPLORATION (On cherche la suite)
        else:
            status = "EXPLORATION / SCAN"
            pydirectinput.keyUp('w')
            move_mouse_hardware(VITESSE_SCAN * direction_exploration)
            # Inversion de scan périodique pour simuler un humain
            if time.time() % 6 > 5:
                direction_exploration *= -1

        # 4. AFFICHAGE INTERFACE DEBUG
        cv2.putText(frame, f"ETAT: {status}", (20, 50), 2, 1, color, 2)
        cv2.putText(frame, f"POINTS SLAM: {nb_points.value}", (20, 90), 2, 0.7, (255, 255, 255), 1)
        cv2.circle(frame, (CENTRE_X, HAUTEUR//2), 6, (0, 0, 255), -1) # "Nez" du bot
        
        cv2.imshow("IA DOORS - SYSTÈME AUTONOME V5", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            pydirectinput.keyUp('w')
            break

cv2.destroyAllWindows()