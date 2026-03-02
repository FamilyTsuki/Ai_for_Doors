from ultralytics import YOLO

from ultralytics import YOLO

if __name__ == '__main__':
    # On repart du modèle de base
    model = YOLO("yolo11n.pt")

    # On lance l'entraînement
    model.train(
        data="data.yaml", 
        epochs=40,      # Un peu plus pour être sûr
        imgsz=320,     # Garde 320 pour ton CPU
        device="cpu", 
        batch=12,       # Petit batch pour ne pas saturer ta RAM
        plots=True     # Pour voir les graphiques de progression
    )