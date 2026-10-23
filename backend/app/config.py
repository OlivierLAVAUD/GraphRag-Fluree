"""Configuration de l'application, pilotée par variables d'environnement."""

import os

# -- Services externes -------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
FLUREE_URL = os.environ.get("FLUREE_URL", "http://fluree:8090")

# -- Graphe Fluree -----------------------------------------------------------
# NB : ne pas réutiliser le nom d'un ledger supprimé (drop) : le serveur garde
# un « tombeau » et toute transaction échoue avec « Ledger has been retracted ».
LEDGER = os.environ.get("FLUREE_LEDGER", "kb/main")

# Préfixe RDF utilisé pour les prédicats du graphe de connaissances
# (https://ns.flurag.dev/content, .../source, .../embedding, .../chunkIndex)
RAG_NS = "https://ns.flurag.dev/"
# Namespace pour les entités nommées (entreprises, personnes, etc.)
ENT_NS = "https://ns.flurag.dev/ent/"
CTX = {"rag": RAG_NS, "ent": ENT_NS}

# -- Modèles Ollama ----------------------------------------------------------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.1:8b")

# -- Paramètres RAG ----------------------------------------------------------
TOP_K = int(os.environ.get("TOP_K", "7"))
# Seuil de similarité : en dessous, un chunk est jugé hors-sujet et n'est pas
# fourni au LLM (réduit le bruit et les réponses inventées à partir d'un
# contexte sans rapport). Ajustable via RETRIEVAL_MIN_SCORE.
RETRIEVAL_MIN_SCORE = float(os.environ.get("RETRIEVAL_MIN_SCORE", "0.3"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))

# -- Limites d'upload ---------------------------------------------------------
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))  # 50 Mo par défaut

# Extensions de fichiers acceptées à l'ingestion
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}