import os

# --- CONFIGURATION ---
# Remplace par le chemin de ton dossier (images ou labels)
dossier = r'./data'
nouveau_nom_base = "lock" # Tu peux mettre "porte" ou "data"
extension_cible = ".png"
# ---------------------

def renommer():
    if not os.path.exists(dossier):
        print(f"Erreur : Le dossier {dossier} n'existe pas.")
        return

    # Liste les fichiers avec la bonne extension
    fichiers = [f for f in os.listdir(dossier) if f.lower().endswith(extension_cible)]
    fichiers.sort() # Trie par ordre alphabétique

    print(f"Fichiers trouvés : {len(fichiers)}")

    for index, nom_original in enumerate(fichiers, start=1):
        ancien_chemin = os.path.join(dossier, nom_original)
        
        # On crée le nouveau nom : image_1.jpg, image_2.jpg...
        nouveau_nom = f"{nouveau_nom_base}_{index}{extension_cible}"
        nouveau_chemin = os.path.join(dossier, nouveau_nom)

        # Renommer physiquement le fichier
        try:
            os.rename(ancien_chemin, nouveau_chemin)
            print(f"Renommé : {nom_original} -> {nouveau_nom}")
        except Exception as e:
            print(f"Erreur sur {nom_original} : {e}")

    print("\nOpération terminée !")

if __name__ == "__main__":
    renommer()