import cv2
import numpy as np
import mss
from ultralytics import YOLO

# 1. Charge ton modèle
model = YOLO("runs/detect/train/weights/best.pt")

# 2. Configuration de la capture
with mss.mss() as sct:
    # On définit la zone de l'écran à capturer (X, Y, Largeur, Hauteur)
    monitor = {"top": 10, "left": 10, "width": 940, "height": 940}

    print("Détection en cours... Appuie sur 'q' pour quitter.")

    while True:
        # Capture l'écran et transforme en tableau numpy
        img = np.array(sct.grab(monitor))
        
        # Conversion BGRA (écran) vers BGR (OpenCV)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 3. Prédiction (on utilise imgsz=320 pour rester fluide sur ton CPU)
        results = model.predict(frame, conf=0.8, imgsz=320, verbose=False)

        # 4. Dessiner les boîtes
        annotated_frame = results[0].plot()

        # 5. Affichage
        cv2.imshow("IA Doors - Vision", annotated_frame)

        # Quitter si on appuie sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()