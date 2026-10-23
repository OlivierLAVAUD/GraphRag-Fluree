# GraphRag-Fluree 🤖◈

Chatbot **RAG** (Retrieval-Augmented Generation) dockerisé, appuyé sur **Fluree** comme graphe
de connaissances et **Ollama** pour l'inférence locale sur **GPU NVIDIA**. Tout tourne en
local : les réponses sont fiables, citées et traçables — la connaissance provient
exclusivement de vos documents.

```
┌────────────┐   HTTP/SSE   ┌──────────────┐       ┌─────────────┐
│ Interface  │ ───────────► │   backend    │ ◄────►│   Ollama    │  (GPU NVIDIA)
│    web     │   (port 8000)│   FastAPI    │       │ llama3.1:8b │
└────────────┘              └──────┬───────┘       │ nomic-embed │
                                   │ JSON-LD       └─────────────┘
                                   ▼
                            ┌─────────────┐
                            │   Fluree    │  graphe de connaissances
                            │  (port 8090)│  (ledger kb/main, versionné)
                            └─────────────┘
```

## Fonctionnalités

- **Ingestion** de documents Markdown, texte brut et **PDF** (découpage en chunks avec
  recouvrement, embeddings, stockage en JSON-LD dans le graphe Fluree).
- **Chat en streaming** avec citations : chaque affirmation renvoie vers sa source
  (fichier, extrait, score de similarité).
- **Graphe de connaissances complet** : Fluree est un ledger immuable et versionné ;
  l'historique de chaque donnée qui alimente les réponses est consultable.
- **100 % local** : aucun appel à un service IA tiers.

## Prérequis

| Élément | Note |
|---|---|
| Docker + Compose v2 | `docker compose version` |
| nvidia-container-toolkit | Nécessaire pour exposer le GPU aux conteneurs |
| Carte NVIDIA ≥ 6 Go VRAM | Testé avec une RTX 3060 (6 Go) |

### Installation de nvidia-container-toolkit (si absent)

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Vérification (optionnelle mais recommandée) :

```bash
docker run --rm --gpus all --device /dev/nvidia0:/dev/nvidia0 nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Démarrage rapide

```bash
docker compose up -d --build
```

Au **premier** démarrage, le service `ollama-init` télécharge les modèles (~5 Go) :

```bash
docker compose logs -f ollama-init     # surveiller le téléchargement
docker compose ps                      # tout doit être Up / healthy
```

Ouvrez ensuite **http://localhost:8000** :

1. Onglet **GPU & modèles** → vérifiez que `llama3.1:8b` apparaît avec une utilisation VRAM
   (le GPU est bien actif).
2. Onglet **Documents** → indexez les fichiers d'exemple fournis (`docs/`) soit via
   l'interface, soit :

   ```bash
   docker compose exec backend python ingest.py docs/
   ```

3. Posez des questions, par exemple :
   - « Quels sont les services Docker de l'application ? »
   - « Comment vérifier que le GPU est utilisé ? »
   - « Mes données partent-elles sur Internet ? »

## Utilisation

### Interface web (http://localhost:8000)

- **Chat** : questions en streaming, réponses citées avec les sources récupérées.
- **Documents** : upload multiple (`.md`, `.txt`, `.pdf`) + rapports d'indexation.
- **GPU & modèles** : état du GPU vu par Ollama et modèles chargés.

### API

| Méthode | Endpoint | Rôle |
|---|---|---|
| `POST` | `/api/chat` | `{"question": "..."}` → réponse en SSE (événements `token`, `sources`, `done`) |
| `POST` | `/api/search` | `{"question": "...", "top_k": 7}` → renvoie les chunks les plus pertinents (sans génération) |
| `POST` | `/api/ingest` | Multipart `files` → indexe des documents dans Fluree |
| `GET`  | `/api/status` | Ledger, modèles, état des services |
| `GET`  | `/api/gpu` | Utilisation GPU / VRAM vue par Ollama |

### CLI d'ingestion

```bash
docker compose exec backend python ingest.py docs/            # dossier récursif
docker compose exec backend python ingest.py docs/faq.md      # un fichier
```

## Évaluer les réponses (jeu de questions pièges)

Un harnais d'évaluation pose 10 questions pièges (multi-hop, négation, pièges
temporels, hallucination…) et compare les réponses aux réponses attendues :

```bash
docker compose exec backend python eval_rag.py
```

- Ingère tous les documents de `docs/` puis génère `eval_results.json`
  (à récupérer avec `docker compose cp backend:/app/eval_results.json .`).
- Pour chaque question : **score 0-1** (scoring sémantique par cosinus sur embeddings, avec fallback lexical)
  + les **extraits récupérés** (source, index, score de similarité, extrait) pour
  distinguer un échec de recherche (*retrieval*) d'un échec de génération.
- Les questions marquées `expected_absence` exigent une information **absente du
  document** : le score vaut 1 si le modèle refuse honnêtement de répondre, 0 s'il
  invente. C'est le test clé contre l'hallucination.
- Jeu de test sur mesure : `tests/eval/qa_france2030.json` (10 questions).

## Configuration (variables d'environnement)

Dans `docker-compose.yml`, service `backend` :

| Variable | Défaut | Rôle |
|---|---|---|
| `CHAT_MODEL` | `llama3.1:8b` | Modèle de génération (voir FAQ pour changer) |
| `EMBED_MODEL` | `nomic-embed-text` | Modèle d'embeddings |
| `TOP_K` | `7` | Nombre de chunks injectés dans le prompt |
| `RETRIEVAL_MIN_SCORE` | `0.3` | Seuil de similarité : en dessous, le chunk est écarté (réduit le bruit et les réponses inventées) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `80` | Découpage des documents |
| `FLUREE_LEDGER` | `kb/main` | Nom du ledger (base) dans Fluree (ne pas réutiliser le nom d'un ledger supprimé : « Ledger has been retracted ») |

## Comment ça marche

1. **Ingestion** : un document est découpé en chunks (~500 caractères, recouvrement 80),
   chaque chunk est embeddé (`nomic-embed-text`, 768 dimensions) et inséré dans Fluree en
   JSON-LD : sujet `rag:Chunk`, prédicats `rag:content`, `rag:embedding`, `rag:source`,
   `rag:chunkIndex`.
2. **Question** : l'embedding de la question est comparé par similarité cosinus aux chunks
   du graphe (index local mis en cache pour éviter le recalcul) ; les `TOP_K` plus proches
   sont retenus après application du seuil `RETRIEVAL_MIN_SCORE`.
3. **Génération** : le prompt contient le contexte récupéré et ordonne au modèle de répondre
   uniquement à partir de celui-ci, en citant les sources `[1]…[k]`. La réponse est
   diffusée en streaming (SSE).

La recherche par similarité cosinus (produit matriciel sur un index local) convient à une base
de quelques milliers de chunks. Pour passer à l'échelle, Fluree propose une recherche
vectorielle dédiée (service `fluree/search-httpd`) — l'architecture du projet est prévue
pour remplacer `app/retriever.py` sans toucher au reste.

## Depannage

**Le GPU n'est pas détecté par Ollama** → installez/configurer nvidia-container-toolkit
(voir Prérequis), puis `docker compose up -d --force-recreate ollama` et vérifiez
`docker compose logs ollama` (doit afficher `inference compute id` / le GPU).

**Erreur de permission sur `fluree-data`** → le conteneur Fluree tourne avec l'UID 1000 :
`sudo chown -R 1000:1000 ./fluree-data` (uniquement si vous montez un dossier hôte).

**Réponse « Je ne dispose pas de cette information… »** → la question sort du champ de la
base : indexez les documents pertinents. Vérifiez aussi `TOP_K` et les scores affichés.

**Mémoire VRAM insuffisante** (OOM) → fermez les autres applications GPU, ou passez à un
modèle plus léger (`phi4-mini:3.8b`) en modifiant `CHAT_MODEL` puis
`docker compose exec backend python -c "import urllib.request; ..." ` (ou redémarrez
`ollama-init` avec le nouveau modèle).

**Tout est vide après `docker compose down -v`** → normal : les volumes (graphe + modèles)
ont été supprimés ; relancez et ré-indexez.

## Sécurité / production

- L'image Fluree supporte l'authentification par token (`did:key` / OIDC) : activez
  `FLUREE_DATA_AUTH_MODE=required` et `FLUREE_ADMIN_AUTH_MODE=required` si le service est
  exposé au-delà de la machine locale.
- Aucune donnée n'est envoyée à l'extérieur ; le déploiement est pensé pour un usage
  local / réseau privé.# GraphRag-Fluree
