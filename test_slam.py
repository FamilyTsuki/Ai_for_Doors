import ctypes
import cv2
import numpy as np
import mss
import os
import time

# --- CONFIGURATION ET CHARGEMENT ---
print("Dossier actuel :", os.getcwd())

slam_lib = None
try:
    print("Tentative de chargement de la DLL...")
    dll_path = os.path.abspath("./slam_lib.dll")
    
    # winmode=0 permet de trouver les DLL OpenCV dans le dossier actuel
    slam_lib = ctypes.CDLL(dll_path, winmode=0)
    
    print("✅ DLL chargée avec succès !")
    
    # Configuration des types :
    # La fonction reçoit (int, int, pointer) et RENVOIE un float (le mouvement moyen)
    slam_lib.process_frame.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte)]
    slam_lib.process_frame.restype = ctypes.c_float

except Exception as e:
    print(f"❌ Erreur critique : {e}")
    slam_lib = None

# --- VARIABLES DE LISSAGE ---
historique_mouvement = []
TAILLE_FILTRE = 5 # Nombre de frames pour la moyenne
start_time = time.time()

# --- BOUCLE PRINCIPALE ---
if slam_lib:
    with mss.mss() as sct:
        # Zone de capture centrale (ajuste selon ta résolution de jeu)
        monitor = {"top": 100, "left": 100, "width": 640, "height": 640}

        print("\n🚀 Analyseur de flux optique actif.")
        print("Observez la valeur de MOUVEMENT.")
        print("Si elle reste proche de 0 pendant que vous avancez, le bot est bloqué.")
        
        while True:
            # 1. Capture d'écran
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 2. Préparation des données pour le C++
            h, w, _ = frame.shape
            data_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

            # 3. Appel au C++ pour obtenir le mouvement brut
            mouvement_brut = slam_lib.process_frame(w, h, data_ptr)

            # 4. Système de lissage (Moyenne glissante)
            historique_mouvement.append(mouvement_brut)
            if len(historique_mouvement) > TAILLE_FILTRE:
                historique_mouvement.pop(0)
            
            mouvement_lisse = sum(historique_mouvement) / len(historique_mouvement)

            # 5. Logique de détection de blocage
            # On laisse 2 secondes au démarrage pour que les points s'initialisent
            temps_ecoule = time.time() - start_time
            
            # Seuil de blocage : 0.08 est souvent une bonne base pour Doors
            if mouvement_lisse < 0.08 and temps_ecoule > 2:
                status = "!!! BLOQUÉ / MUR !!!"
                color = (0, 0, 255) # Rouge
            else:
                status = f"MOUVEMENT: {mouvement_lisse:.2f}"
                color = (0, 255, 0) # Vert

            # 6. Affichage HUD pour le debug
            cv2.rectangle(frame, (10, 20), (350, 70), (0, 0, 0), -1) # Fond noir pour lisibilité
            cv2.putText(frame, status, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow("SLAM Motion Analyzer", frame)

            # Quitter avec 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
else:
    print("\n[STOP] Impossible de démarrer. Vérifiez que toutes les DLL sont présentes.")