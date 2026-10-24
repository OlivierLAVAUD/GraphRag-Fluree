# Journal des modifications UI

## 2026-10-24 — Base de connaissances unifiée

### Ajouté

- Modale « Base de connaissances » organisée en cinq onglets : Documents, Recherche, Entités, Historique et GPU.
- Liste des sources indexées avec leur nombre de chunks, alimentée par `GET /api/sources`.
- Recherche dans les chunks avec `top_k`, filtre source, filtre entité et seuil de score.
- Envoi direct d'un résultat de recherche vers le chat, avec reprise des filtres sélectionnés.
- Grille des entités du graphe avec leur type et leur fréquence, alimentée par `GET /api/entities`.
- Chronologie des commits Fluree avec identifiant, date et volumes d'assertions/retraits.
- Filtres de conversation dans le composer : source, entité et `top_k`.
- Routes backend `GET /api/sources` et `GET /api/entities`.
- Champs `source`, `entity` et `top_k` dans la requête `POST /api/chat`.

### Modifié

- L'ancienne modale Documents/GPU devient la modale Base de connaissances ; les raccourcis Documents, Modèles et GPU ouvrent désormais l'onglet correspondant.
- Le chat transmet au retriever les filtres source, entité et `top_k` choisis dans le composer.
- Le seuil de score de la recherche est appliqué côté interface ; la recherche backend conserve son `min_score` de configuration.
- L'indexation met à jour la liste des sources et des entités après ingestion.

### Vérifications

- `pytest` : 49 tests réussis.
- `black --check` et `flake8` : conformes.
- Routes `/api/sources`, `/api/entities`, `/api/search` et `/api/chat` vérifiées avec les filtres.
- Parcours interface vérifié avec un DOM simulé : onglets, recherche, envoi au chat, entités et historique.
