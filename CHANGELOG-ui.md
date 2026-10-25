# Journal des modifications UI

## 2026-10-25 — Sources repliées sous la réponse

### Ajouté

- La barre « Sources » de chaque réponse est un bouton : un clic la déplie ou la replie (`aria-expanded`, chevron, accessible au clavier).

### Modifié

- Les sources sont repliées par défaut : la réponse reste lisible, la barre affiche le nombre d'extraits et le seuil de similarité appliqué.

### Vérifications

- `node --check app.js` conforme ; `pytest` : 105 tests réussis.
- Chromium headless sur la pile Docker, flux `/api/chat` simulé : bloc replié au départ (`listVisible: false`, `aria-expanded="false"`, chevron droit), déplié au clic (`aria-expanded="true"`, 2 cartes visibles), replié au second clic ; la carte d'extrait ouvre toujours la modale.

## 2026-10-25 — Ingestion d'un dossier et progression

### Ajouté

- Bouton « Choisir un dossier » : indexation d'un dossier entier (sous-dossiers compris), en plus de la sélection de fichiers.
- Barre de progression de l'ingestion : fichier en cours, avancement (`3/12 · 42 %`), rayures animées pendant le traitement, puis bilan (`Terminé · 12 fichier(s) indexé(s)` ou nombre d'échecs).
- Bilan par fichier, y compris les échecs (`fichier — échec : …`), sans interrompre les fichiers suivants.

### Modifié

- La zone de dépôt affiche le nom du document ou du dossier sélectionné (et son poids) au lieu de « Sélectionner des documents » ; pendant l'ingestion, elle affiche le fichier en cours.
- L'ingestion envoie un fichier par requête : la progression reflète le travail réel (embeddings + écriture Fluree) et un fichier en erreur n'annule pas les autres.
- Les formats non supportés d'un dossier sélectionné sont filtrés côté interface, avec le nombre de fichiers ignorés.
- `GET /api/status` expose `limits` (nombre de fichiers, tailles, extensions) : l'interface refuse une sélection hors limites avant l'envoi.
- `POST /api/ingest` conserve le chemin relatif d'un upload de dossier comme source (`mon-dossier/sous/notes.md`, segments `..` retirés).

### Vérifications

- `pytest` : 105 tests réussis (dont 4 sur les chemins relatifs et les limites exposées).
- `black --check`, `flake8`, `node --check app.js` : conformes.
- Chromium headless sur la pile Docker, réponses `/api/ingest` simulées : sélection d'un dossier de 2 fichiers → titre « mon-dossier », libellé « Dossier · 2 fichier(s) · 32 o », nom du fichier affiché pendant l'ingestion, barre à 100 %, « Terminé · 2 fichier(s) indexé(s) » ; en cas de refus serveur, une ligne d'échec par fichier et « 0/2 fichier(s) indexé(s), 2 en échec ».

## 2026-10-25 — Désindexation et réinitialisation

### Ajouté

- Bouton de désindexation (`×`) sur chaque source de l'onglet Documents, avec confirmation et rafraîchissement des sources, des entités et de l'état des services.
- Zone « Réinitialiser la base » en bas de l'onglet Documents, avec confirmation et remise à zéro de la liste des sources.
- Routes backend `DELETE /api/sources?source=…` et `POST /api/reset` (désindexation logique : chunks marqués obsolètes, ledger conservé).

### Modifié

- Les lignes de la liste des sources affichent le nom tronqué avec une info-bulle, le nombre de chunks et le bouton de suppression.

### Corrigé

- `/` et `/static/*` sont servis avec `Cache-Control: no-cache` : après un `docker compose up --build`, le navigateur ne peut plus afficher une UI périmée (boutons absents). La revalidation par ETag reste active.

### Vérifications

- `pytest` : 101 tests réussis (dont 9 sur la désindexation et la réinitialisation, 2 sur les en-têtes de cache et le contenu de la page servie).
- `black --check` et `flake8` : conformes ; `node --check app.js` : conforme.
- Sélecteurs `#resetKb` / `#sourceList` vérifiés dans `index.html` (75 sélecteurs `#id` d'`app.js` résolus).
- Rendu réel dans Chromium headless sur la pile Docker : la source indexée apparaît avec son bouton `×` et la zone « Réinitialiser la base » est présente.

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
