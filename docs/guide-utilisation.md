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

- **Interface web** : ouvrez http://localhost:8000, onglet « Documents ». La zone de dépôt accepte plusieurs fichiers (.md, .markdown, .txt, .pdf) ; le bouton « Choisir un dossier » indexe d'un coup tous les documents d'un dossier, sous-dossiers compris. Le nom du document (ou du dossier) sélectionné s'affiche dans la zone de dépôt, et une barre de progression suit l'indexation fichier par fichier. Cliquez ensuite sur « Indexer ».

  Un dossier uploadé conserve son chemin relatif comme source (`mon-dossier/sous/notes.md`) : deux fichiers de même nom dans des dossiers différents restent distincts. Seuls les formats supportés sont retenus, les autres fichiers du dossier sont ignorés (le nombre est signalé).
- **CLI** : déposez vos fichiers dans `docs/` puis exécutez :
  - `docker compose exec backend python ingest.py docs/`

L'indexation est idempotente : ré-ingérer le même fichier ne duplique pas la connaissance (les doublons sont éliminés à la recherche).

## Désindexer un document ou réinitialiser la base

Tout se fait dans l'onglet **Documents** de la modale « Base de connaissances » (carte en bas de la barre latérale, ou pastille « Documents » sous le champ de saisie, ou menu « ⋯ » → *Importer des documents*) :

- **Un document** : bouton `×` à droite de chaque source de la liste « Sources indexées ».
- **Tous les documents** : zone rouge « Réinitialiser la base », en bas de l'onglet.

La désindexation est logique : les chunks sont marqués obsolètes, disparaissent des réponses, mais le ledger Fluree et son historique sont conservés (ré-indexer le fichier le réactive). Pour repartir d'un ledger réellement vide, arrêtez la pile et supprimez le volume : `docker compose down -v`.

Si les boutons n'apparaissent pas après une mise à jour, videz le cache du navigateur (Ctrl+Maj+R) : le serveur envoie désormais `Cache-Control: no-cache` sur la page et les fichiers de l'interface.

## Interroger le chatbot

Ouvrez http://localhost:8000, posez votre question dans le champ en bas. La réponse arrive en streaming, accompagnée des sources (fichier + numéro d'extrait + score de similarité).

Les sources sont repliées par défaut sous la réponse : cliquez sur la barre « Sources » (nombre d'extraits et seuil de similarité) pour les déplier, puis sur un extrait pour lire le passage complet.

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