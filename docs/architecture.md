# Architecture de l'application

L'application est une solution RAG (Retrieval-Augmented Generation) complète exécutée localement, entièrement conteneurisée avec Docker Compose. Aucune donnée ne quitte la machine : les modèles d'IA tournent sur le GPU NVIDIA grâce à Ollama.

## Services Docker

Le fichier docker-compose.yml orchestre quatre services :

1. **ollama** : serveur de modèles locaux (LLM de génération + modèle d'embeddings). Il utilise le GPU NVIDIA via nvidia-container-toolkit. Les modèles sont persistés dans le volume `ollama-models`.
2. **ollama-init** : service jetable qui télécharge les modèles au premier démarrage : `llama3.1:8b` pour la génération et `nomic-embed-text` pour les embeddings.
3. **fluree** : le graphe de connaissances (Fluree Server 1.0), exposé sur le port 8090. Les données sont persistées dans le volume `fluree-data`. Chaque commit est immuable et horodaté (historique complet).
4. **backend** : API FastAPI (port 8000) qui orchestre le pipeline : ingestion des documents, recherche sémantique dans Fluree et génération des réponses.

## Pipeline RAG

- **Ingestion** : les documents (Markdown, texte, PDF) sont découpés en chunks d'environ 500 caractères avec recouvrement de 80 caractères, puis embeddés en vecteurs de 768 dimensions par `nomic-embed-text`.
- **Stockage** : chaque chunk est inséré dans le ledger `kb/main` de Fluree en JSON-LD, avec ses métadonnées (contenu, embedding, source, index). Le graphe assure la traçabilité complète des modifications grâce à son modèle de ledger immuable.
- **Recherche** : à chaque question, l'embedding de la question est comparé (similarité cosinus) aux embeddings des chunks stockés ; les 5 plus proches sont retenus.
- **Génération** : le prompt embarque le contexte récupéré avec ses sources, et `llama3.1:8b` génère une réponse citée, en français par défaut.

## Choix techniques

- Python + FastAPI : écosystème IA mature, API asynchrone simple.
- Fluree : graphe de connaissances versionné (ledger), idéal pour auditer quelles données alimentent les réponses.
- Ollama : inférence locale sur GPU, aucun appel réseau vers un service tiers.

## Ports exposés

| Port | Service      | Usage                                   |
|------|--------------|-----------------------------------------|
| 8000 | backend      | Interface web et API du chatbot         |
| 8090 | fluree       | API HTTP de Fluree                      |
| 11434| ollama       | API HTTP de Ollama                      |