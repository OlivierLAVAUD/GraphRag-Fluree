# FAQ

## Quels modèles d'IA sont utilisés ?

Deux modèles locaux via Ollama :
- **llama3.2:3b** pour la génération des réponses (quantifié Q4, environ 2,6 Go en VRAM).
- **nomic-embed-text** pour transformer les textes en vecteurs de 768 dimensions (environ 0,3 Go).

## Mes données partent-elles sur Internet ?

Non. L'inférence, les embeddings et le stockage s'exécutent entièrement en local (GPU NVIDIA + volumes Docker). Aucun appel vers un service d'IA externe n'est effectué. Seules des mises à jour d'images Docker au démarrage utilisent le réseau.

## Pourquoi un graphe de connaissances plutôt qu'une simple base vectorielle ?

Fluree est un graphe versionné (ledger) : chaque insertion est horodatée et l'historique complet est conservé. L'application l'expose via `GET /api/history` ; le serveur Fluree fournit aussi ses propres API d'historique (`/log`, `/show`). On peut donc auditer exactement quelles données alimentent le RAG, vérifier leur provenance et garantir des réponses traçables et fiables. De plus, le modèle de données RDF/JSON-LD permet d'enrichir la connaissance avec des relations sémantiques entre entités.

## Comment ajouter de nouvelles connaissances ?

Déposez des fichiers Markdown, texte ou PDF via l'interface web ou la CLI d'ingestion. Toute la connaissance répondue par le chatbot provient exclusivement de ces documents : le modèle refuse de répondre si l'information est absente du contexte récupéré.

## Comment réinitialiser entièrement la base de connaissances ?

1. `docker compose down -v` pour supprimer les volumes, puis `docker compose up -d` pour repartir de zéro.
2. Ou changez de ledger en définissant la variable `FLUREE_LEDGER` du service backend (ex: `kb/main2`), puis redémarrez : `docker compose up -d backend`.

⚠️ **Ne réutilisez pas le nom d'un ledger supprimé** : après un `drop`, le serveur garde un « tombeau » et toute transaction échoue avec `Ledger has been retracted`, même après redémarrage. Utilisez un nouveau nom de ledger (variable `FLUREE_LEDGER`).

## La carte graphique est-elle suffisante ?

Une RTX 3060 (6 Go de VRAM) est suffisante : llama3.2:3b en Q4 occupe ~2,6 Go et nomic-embed-text ~0,3 Go, soit les deux modèles chargés simultanément sur le GPU. Un modèle 8B (~4,9 Go) ne tient pas avec le modèle d'embeddings et une partie du calcul bascule sur CPU : la génération devient très lente, avec des rechargements de plusieurs minutes. Évitez aussi d'avoir d'autres applications GPU ouvertes.

## Puis-je changer de modèle ?

Oui : modifiez les variables `CHAT_MODEL` et `EMBED_MODEL` du service `backend` dans docker-compose.yml, puis `ollama pull <modele>` via le service ollama-init. Le modèle d'embedding détermine la dimension des vecteurs : changer d'embedding impose de ré-indexer les documents.