from ultralytics import YOLO
import cv2

# 1. Charger le modèle (il se télécharge au premier lancement)
model = YOLO('yolov8n.pt') 

# 2. Exécuter la détection sur une image de test
results = model('https://ultralytics.com/images/bus.jpg')

# 3. Afficher les résultats
for result in results:
    result.show()  # Ouvre une fenêtre avec l'image annotée

