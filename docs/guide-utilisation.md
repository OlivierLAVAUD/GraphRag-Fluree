# Guide d'utilisation

## Prérequis

- Docker avec le plugin Compose v2.
- nvidia-container-toolkit installé et configuré (nécessaire pour exposer le GPU NVIDIA aux conteneurs).
- Une carte NVIDIA avec au moins 6 Go de VRAM (une RTX 3060 convient).

## Démarrage

1. Lancez toute la pile : `docker compose up -d --build`
2. Au premier démarrage, le service `ollama-init` télécharge les modèles (environ 5 Go) : surveillez avec `docker compose logs -f ollama-init`.
3. Vérifiez que tout est sain : `docker compose ps` (tous les services doivent être "healthy" ou "Up").

## Indexer des documents

Deux méthodes :

- **Interface web** : ouvrez http://localhost:8000, onglet « Documents », sélectionnez vos fichiers (.md, .markdown, .txt, .pdf) et cliquez sur « Indexer les fichiers ».
- **CLI** : déposez vos fichiers dans `docs/` puis exécutez :
  - `docker compose exec backend python ingest.py docs/`

L'indexation est idempotente : ré-ingérer le même fichier ne duplique pas la connaissance (les doublons sont éliminés à la recherche).

## Interroger le chatbot

Ouvrez http://localhost:8000, posez votre question dans le champ en bas. La réponse arrive en streaming, accompagnée des sources (fichier + numéro d'extrait + score de similarité).

Exemples de questions avec les documents d'exemple :
- « Quels sont les services Docker de l'application ? »
- « Comment indexe-t-on des documents ? »
- « Quelle carte NVIDIA est recommandée ? »

## Vérifier que le GPU est utilisé

Onglet « GPU & modèles » de l'interface : chaque modèle chargé affiche sa consommation en VRAM et le nom du GPU (ex : NVIDIA GeForce RTX 3060). Si aucun GPU n'apparaît, consultez la section « Dépannage » du README.

## Arrêt et données

- `docker compose down` arrête les services sans supprimer les données.
- `docker compose down -v` supprime aussi les volumes (données Fluree et modèles Ollama à retélécharger).
- Les données (documents indexés et historique du graphe) sont persistées dans les volumes Docker `fluree-data` et `ollama-models`.