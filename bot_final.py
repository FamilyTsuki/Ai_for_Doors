import ctypes
import cv2
import numpy as np
import mss
import os
import time
import math
import random
import pydirectinput
from ultralytics import YOLO
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# 1. ARCHITECTURE SYSTÈME ET PARAMÈTRES CRITIQUES (LIGNES 20-100)
# ==============================================================================
MOUSEEVENTF_MOVE = 0x0001
WIDTH, HEIGHT = 1200, 940
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
PATH_MODEL = "runs/detect/train/weights/best.pt"
DLL_SLAM = os.path.abspath("./slam_lib.dll")

@dataclass
class PhysicsEngine:
    """ Gère la dynamique de mouvement de l'agent """
    velocity: np.array = field(default_factory=lambda: np.array([0.0, 0.0]))
    acceleration: float = 0.45
    friction: float = 0.82
    rotation_speed: float = 0.28
    look_ahead_factor: float = 1.4

@dataclass
class CognitiveState:
    """ Mémoire spatiale et hiérarchie des tâches """
    objective_stack: list = field(default_factory=list)
    last_known_door: np.array = field(default_factory=lambda: np.array([CENTER_X, CENTER_Y]))
    exploration_map: np.array = field(default_factory=lambda: np.zeros((10, 10))) # Mini-grid
    stuck_buffer: deque = field(default_factory=lambda: deque(maxlen=30))
    frame_time: float = time.time()

# ==============================================================================
# 2. MOTEUR DE NAVIGATION PAR CHAMPS DE FORCE (LIGNES 101-200)
# ==============================================================================
class NavigationPro:
    @staticmethod
    def get_repulsion_vector(frame):
        """ 
        Analyse le flux optique pour éviter les collisions murales.
        
        """
        # Analyse des zones de danger (Bords de l'écran)
        left_slice = frame[HEIGHT//3:, :WIDTH//6]
        right_slice = frame[HEIGHT//3:, 5*WIDTH//6:]
        
        # Calcul de la "densité d'obstacle" par gradient de Sobel
        l_grad = cv2.Sobel(cv2.cvtColor(left_slice, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=5)
        r_grad = cv2.Sobel(cv2.cvtColor(right_slice, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=5)
        
        l_force = np.mean(np.abs(l_grad))
        r_force = np.mean(np.abs(r_grad))
        
        # Force résultante de répulsion
        repulsion_x = 0
        if l_force > 15: repulsion_x += (l_force * 0.8)
        if r_force > 15: repulsion_x -= (r_force * 0.8)
        return repulsion_x

# ==============================================================================
# 3. CONTRÔLEUR DE MOUVEMENT BALISTIQUE (LIGNES 201-300)
# ==============================================================================
class BallisticInput:
    @staticmethod
    @staticmethod
    def move_smooth(dx, dy=0):
        """ 
        Mouvement de caméra haute fréquence utilisant une courbe sigmoïde 
        pour l'accélération et un amortissement quadratique pour la précision.
        """
        if abs(dx) < 1.5: return # Ignore les micro-bruits
        
        # 1. CALCUL DE LA COURBE SIGMOÏDE
        # Elle permet de démarrer lentement, d'accélérer brusquement au milieu, 
        # et de ralentir à l'approche de la cible.
        # 



        
        # Le facteur 100 détermine la 'pente' de l'accélération
        speed_multiplier = 1 / (1 + math.exp(-abs(dx) / 100))
        
        # 2. AMORTISSEMENT DYNAMIQUE (DAMPING)
        # On réduit la puissance si l'erreur est petite pour éviter l'overshoot
        # 
        damping = min(1.0, abs(dx) / 150)
        
        # 3. CALCUL DU DELTA FINAL
        # On combine la sigmoïde, l'amortissement et le gain de rotation (0.5)
        final_dx = int(dx * speed_multiplier * damping * 0.6)
        
        # 4. INJECTION HARDWARE
        # On limite le mouvement par frame pour ne pas 'casser' le moteur de Roblox
        if abs(final_dx) > 200: final_dx = 200 if final_dx > 0 else -200
        
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, final_dx, int(dy), 0, 0)
    @staticmethod
    def urgent_interact():
        """ Interaction turbo sans délai """
        pydirectinput.keyDown('e')
        time.sleep(0.04) # Timing minimal pour le serveur Roblox
        pydirectinput.keyUp('e')

# ==============================================================================
# 4. COUCHE DE PERCEPTION MULTI-THREADÉE (LIGNES 301-450)
# ==============================================================================
class GodPerception:
    def __init__(self):
        self.model = YOLO(PATH_MODEL)
        self.slam = ctypes.CDLL(DLL_SLAM, winmode=0)
        self.slam.process_frame.argtypes = [ctypes.c_int, ctypes.c_int, 
                                          ctypes.POINTER(ctypes.c_ubyte), 
                                          ctypes.POINTER(ctypes.c_int)]
        self.slam.process_frame.restype = ctypes.c_float
        self.executor = ThreadPoolExecutor(max_workers=2)

    def process(self, frame):
        # Lancement parallèle : YOLO et SLAM
        future_yolo = self.executor.submit(self.model.predict, frame, conf=0.5, imgsz=320, verbose=False)
        
        nb_pts = ctypes.c_int(0)
        ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        motion = self.slam.process_frame(WIDTH, HEIGHT, ptr, ctypes.byref(nb_pts))
        
        results = future_yolo.result()
        return results[0].boxes, motion, nb_pts.value

# ==============================================================================
# 5. AGENT DÉCISIONNEL SUPRÊME (LIGNES 451-600)
# ==============================================================================
class GodAgent:
    def __init__(self):
        self.perception = GodPerception()
        self.physics = PhysicsEngine()
        self.cog = CognitiveState()

    def run_cycle(self, frame):
        # 1. Perception
        boxes, motion, pts = self.perception.process(frame)
        
        # 2. Analyse des forces de navigation
        repulsion_x = NavigationPro.get_repulsion_vector(frame)
        
        # 3. Analyse des objectifs (Priorité dynamique)
        target_x = None
        best_box = None
        
        if len(boxes) > 0:
            # On cherche l'objectif le plus "rentable" (Taille * Priorité / Distance)
            priorities = {'key': 5, 'lever': 4, 'door': 2, 'drawer': 1}
            scored_objs = []
            for b in boxes:
                label = self.perception.model.names[int(b.cls[0])]
                bx = b.xyxy[0]
                center_obj = (bx[0] + bx[2]) / 2
                score = (priorities.get(label, 0) * (bx[2]-bx[0])) / (abs(center_obj - CENTER_X) + 1)
                scored_objs.append((score, center_obj, b))
            
            scored_objs.sort(key=lambda x: x[0], reverse=True)
            target_x = scored_objs[0][1]
            best_box = scored_objs[0][2]

        # 4. Exécution du pilotage
        if target_x:
            # Calcul de l'erreur + Anticipation
            error = (target_x - CENTER_X) + repulsion_x
            BallisticInput.move_smooth(error)
            
            # Marche et Strafe
            pydirectinput.keyDown('w')
            if error > 100: pydirectinput.keyDown('d'); pydirectinput.keyUp('a')
            elif error < -100: pydirectinput.keyDown('a'); pydirectinput.keyUp('d')
            else: pydirectinput.keyUp('a'); pydirectinput.keyUp('d')
            
            # Interaction automatique
            if (best_box.xyxy[0][3] - best_box.xyxy[0][1]) > (HEIGHT * 0.45):
                BallisticInput.urgent_interact()
            
            self.cog.stuck_buffer.append(motion)
            status = "PURSUIT_ACTIVE"
        
        # 5. Gestion de l'échec de progression
        elif len(self.cog.stuck_buffer) == 30 and np.mean(self.cog.stuck_buffer) < 0.01:
            status = "EMERGENCY_RECOVERY"
            pydirectinput.keyUp('w')
            pydirectinput.press('s', presses=2)
            BallisticInput.move_smooth(500)
            self.cog.stuck_buffer.clear()
        
        else:
            status = "OPTIMIZED_SCAN"
            scan = math.sin(time.time() * 3) * 20
            BallisticInput.move_smooth(scan)
            if int(time.time() * 2) % 4 == 0: pydirectinput.press('w')

        return status

# ==============================================================================
# 6. INITIALISATION ET BOUCLE INFINIE
# ==============================================================================
agent = GodAgent()
with mss.mss() as sct:
    mon = {"top": 10, "left": 10, "width": WIDTH, "height": HEIGHT}
    print("⚡ IA DOORS V14 : PROTOCOLE SUPRÊME ACTIVÉ")
    
    while True:
        loop_start = time.time()
        frame = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)
        
        state = agent.run_cycle(frame)
        
        # HUD Debug minimaliste (pour la performance)
        cv2.putText(frame, state, (20, 40), 1, 1.5, (0, 255, 0), 2)
        cv2.imshow("IA DOORS GOD-MODE", frame)

        dt = time.time() - loop_start
        if cv2.waitKey(max(1, int((0.016 - dt)*1000))) & 0xFF == ord('q'):
            pydirectinput.keyUp('w')
            break