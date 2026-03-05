import ctypes
import cv2
import numpy as np
import mss
import os
import time

# --- INITIALISATION DLL ---
dll_path = os.path.abspath("./slam_lib.dll")
try:
    slam_lib = ctypes.CDLL(dll_path, winmode=0)
    slam_lib.process_frame.argtypes = [
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.POINTER(ctypes.c_ubyte), 
        ctypes.POINTER(ctypes.c_int)
    ]
    slam_lib.process_frame.restype = ctypes.c_float
    print("✅ Moteur SLAM chargé.")
except Exception as e:
    print(f"❌ Erreur chargement DLL : {e}")
    exit()

# --- VARIABLES ---
historique_mouvement = []
nb_points = ctypes.c_int(0)
start_time = time.time()

with mss.mss() as sct:
    # Zone de capture
    monitor = {"top": 100, "left": 100, "width": 640, "height": 640}
    
    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w, _ = frame.shape
        data_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        
        # Appel C++ (on passe nb_points par référence)
        mouvement_brut = slam_lib.process_frame(w, h, data_ptr, ctypes.byref(nb_points))

        # Lissage du mouvement (moyenne sur 5 frames)
        historique_mouvement.append(mouvement_brut)
        if len(historique_mouvement) > 5: historique_mouvement.pop(0)
        mouvement_lisse = sum(historique_mouvement) / len(historique_mouvement)

        # Logique d'alerte
        # Si on a des points mais qu'ils ne bougent pas = BLOQUÉ
        # Si on n'a plus de points = AVEUGLE
        temps_total = time.time() - start_time
        if nb_points.value < 10 and temps_total > 2:
            status = "AVEUGLE (Pas de texture)"
            color = (0, 165, 255) # Orange
        elif mouvement_lisse < 0.08 and temps_total > 2:
            status = "BLOQUÉ (Mur détecté)"
            color = (0, 0, 255) # Rouge
        else:
            status = f"Mouvement: {mouvement_lisse:.2f}"
            color = (0, 255, 0) # Vert

        # Affichage HUD
        cv2.rectangle(frame, (5, 5), (300, 100), (0,0,0), -1)
        cv2.putText(frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Points actifs: {nb_points.value}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("SLAM Doors - Pro Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()