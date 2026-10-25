"""Configuration de l'application, pilotée par variables d'environnement."""

import os
import tempfile

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
# Un modèle 3B tient entièrement sur un GPU 6 Go avec le modèle d'embeddings ;
# un 8B déborde, bascule partiellement sur CPU et rend la génération inutilisable.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2:3b")

# -- Paramètres RAG ----------------------------------------------------------
TOP_K = int(os.environ.get("TOP_K", "7"))
# Seuil de similarité : en dessous, un chunk est jugé hors-sujet et n'est pas
# fourni au LLM (réduit le bruit et les réponses inventées à partir d'un
# contexte sans rapport). Ajustable via RETRIEVAL_MIN_SCORE.
RETRIEVAL_MIN_SCORE = float(os.environ.get("RETRIEVAL_MIN_SCORE", "0.3"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))

# -- Cache de l'index de recherche -------------------------------------------
# Marqueur partagé entre processus (uvicorn et CLI `ingest.py`) : il change à
# chaque ingestion pour que le backend reconstruise son index local sans
# redémarrage. Placé dans /tmp : lisible par tous les processus du conteneur.
INDEX_VERSION_FILE = os.environ.get(
    "INDEX_VERSION_FILE",
    os.path.join(tempfile.gettempdir(), "graphrag-index.version"),
)
# Cache disque de l'index (matrice + métadonnées) : évite de rapatrier tous les
# embeddings depuis Fluree à chaque redémarrage du processus.
INDEX_CACHE_DIR = os.environ.get(
    "INDEX_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "graphrag-index-cache"),
)
# Nombre maximal de fichiers d'index conservés sur disque (un par version).
INDEX_CACHE_MAX_ENTRIES = int(os.environ.get("INDEX_CACHE_MAX_ENTRIES", "32"))
# Budget mémoire du cache d'index (matrice + métadonnées). Au-delà, l'index
# n'est pas gardé en mémoire : la recherche reste correcte mais reconstruit
# la matrice à chaque requête plutôt que de faire grossir le processus.
# 0 désactive complètement le cache mémoire (le cache disque reste actif).
INDEX_CACHE_MAX_MB = int(os.environ.get("INDEX_CACHE_MAX_MB", "256"))

# -- Limites d'upload ---------------------------------------------------------
MAX_UPLOAD_SIZE = int(
    os.environ.get("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024))
)  # 50 Mo par fichier par défaut
MAX_TOTAL_UPLOAD_SIZE = int(
    os.environ.get("MAX_TOTAL_UPLOAD_SIZE", str(200 * 1024 * 1024))
)  # 200 Mo par requête
MAX_UPLOAD_FILES = int(os.environ.get("MAX_UPLOAD_FILES", "20"))

# -- Limites de l'API ---------------------------------------------------------
MAX_QUESTION_LENGTH = int(os.environ.get("MAX_QUESTION_LENGTH", "4000"))
MAX_TOP_K = int(os.environ.get("MAX_TOP_K", "50"))
# Jeton optionnel : s'il est défini, les endpoints d'écriture/chat exigent
# l'en-tête `X-API-Token` (ou `Authorization: Bearer <jeton>`).
API_TOKEN = os.environ.get("API_TOKEN", "")
# Générations simultanées autorisées (protège le GPU).
MAX_CONCURRENT_CHATS = int(os.environ.get("MAX_CONCURRENT_CHATS", "2"))

# Extensions de fichiers acceptées à l'ingestion
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}
