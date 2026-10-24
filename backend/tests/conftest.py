"""Isolation des caches d'index entre tests.

Le marqueur de version et le cache disque sont globaux au processus : sans
isolation, un test pourrait réutiliser l'index d'un autre test (ou d'une
exécution précédente). Chaque test travaille donc dans un répertoire dédié.
"""

import pytest

from app import config, retriever


@pytest.fixture(autouse=True)
def isolate_index_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_VERSION_FILE", str(tmp_path / "index.version"))
    monkeypatch.setattr(config, "INDEX_CACHE_DIR", str(tmp_path / "index-cache"))
    retriever.invalidate_index()
    yield
    retriever.invalidate_index()
