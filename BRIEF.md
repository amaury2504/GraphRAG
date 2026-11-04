# Brief de reprise (pour la prochaine session Claude)

À lire après `Claude.md` et `Spec.md`. État au 2026-11-03. Ce fichier résume ce qui est fait, ce qui a été décidé, et les pièges déjà rencontrés, pour ne pas les redécouvrir.

## 1. Où on en est

| Étape | État |
| --- | --- |
| 1. Socle (structure, Docker Compose, CI) | **Terminée**, CI verte |
| 2. Ingestion (arXiv, auteurs, citations → Neo4j) | **Terminée**, corpus complet ingéré le 2026-11-04. |
| 3. RAG de base | **Commencée** : découpage des résumés et port `Embedder` faits. En attente : choix du modèle d'embeddings et du LLM. |

État de la base locale (2026-11-04) : 2 827 articles, 10 383 auteurs, 11 203 `AUTHORED_BY`, 924 `CITES` (539 articles citent, 526 sont cités). Semantic Scholar : 2 324 articles trouvés, 3 inconnus, 93 sans référence arXiv.

## 2. Décisions prises (ne pas les rouvrir sans raison)

- **Python 3.14.2**, `uv` (versions figées dans `uv.lock`), `pytest`, `ruff`, `mypy --strict`.
- **Corpus « A »** : catégories cs.CL/IR/AI/LG, résumé contenant « retrieval-augmented » (ou « retrieval augmented »), 12 mois. 2 829 articles comptés le 2026-11-03. Titres et résumés seulement.
- **Auteurs : OpenAlex**, sans clé, par DOI arXiv (`10.48550/arXiv.<id>`), lots de 100. Créés uniquement en phase 2, avec leurs ids OpenAlex. Un article inconnu d'OpenAlex n'a pas de nœud `Auteur` (environ 20 % sur le lot).
- **Citations : Semantic Scholar**, sans clé, `POST /graph/v1/paper/batch`, lots de 100. OpenAlex a été écarté pour les citations : il renvoie 0 référence pour la plupart des articles arXiv. `CITES` seulement entre articles du corpus.
- **Reprise des citations** (2026-11-03) : fichier local `data/citations_checked.txt` (ignoré par git, rien dans le graphe), un id par ligne, marqué après l'écriture des `CITES`. `--citations-only` saute arXiv/OpenAlex, `--recheck-citations` ignore le fichier (à utiliser si Neo4j a été vidé). Un lot dont les essais sont épuisés est sauté ; code de sortie 3 = relancer plus tard.
- **Clés d'unicité** : `Article.arxiv_id` (sans version), `Author.key` (id OpenAlex). Écriture par `MERGE` en lots de 500. Un article déjà présent n'est jamais modifié (`ON CREATE SET`).

## 3. Décisions encore ouvertes (à proposer en deux options, jamais à trancher seul)

- **Version de Neo4j** : 5.26.32 LTS est en place **à titre provisoire**. Alternative : 2026.10. Vérifier dans la doc officielle le support des index vectoriels et texte intégral avant l'étape 3.
- **Modèle d'embeddings** (et de reranking), **fournisseur de LLM**, **budget d'appels**.
- **Fusion de la recherche hybride** : décidé le 2026-11-03, option A (Reciprocal Rank Fusion, `1/(60 + rang)`, fusion en Python dans `pipeline/`). **Option B à essayer après l'étape 4** : somme pondérée de scores normalisés (min-max, poids à régler sur le jeu de test), si le jeu de test montre un écart.
- Normalisation des noms de méthodes/benchmarks (avant l'étape 6), taille du lot d'extraction, nombre de testeurs, seuils de latence et de coût.
- Licence du dépôt (aucun fichier `LICENSE` pour l'instant).
- **Densité du graphe de citations** (à trancher à l'étape de l'élargissement par le graphe) : 924 `CITES` pour 2 827 articles, car un lien n'existe que si les deux articles sont dans la fenêtre de 12 mois. Option A : garder les articles cités hors corpus comme nœuds minces (id + titre, sans résumé). Option B : élargir la fenêtre (par exemple 24 mois).
- **Doublons arXiv** (à trancher à l'étape 3) : un même travail déposé deux fois sur arXiv donne deux nœuds `Article`, par exemple `2608.12129` et `2606.15971`. Semantic Scholar les signale comme alias, dans le log de l'ingestion (`arXiv alias: ...`) ; d'autres existent sans doute sans être signalés.

## 4. Préférences de l'utilisateur (importantes)

- Échanges en **français**, code et messages de commit en **anglais**.
- **Très rigoureux sur les tests** : tests unitaires, tests d'intégration sur le vrai Neo4j, et tests de mutation ponctuels (casser volontairement le code pour vérifier que les tests échouent).
- **Aucune ligne `Co-Authored-By` ni mention de Claude** dans les commits ou les PR (demande explicite).
- **Petits commits, un sujet par commit, après test**, puis `git push origin main` (pas de force push sans accord ; le contrôle de permissions de Claude Code bloque `--force`).
- **Rien de payant sans accord préalable.** L'ingestion actuelle est gratuite (arXiv, OpenAlex, Semantic Scholar anonymes).
- Une seule étape à la fois : à la fin, résumer et **attendre le feu vert**. Avant de coder la recherche hybride, le schéma du graphe/requêtes d'élargissement ou le jeu de test, **présenter la conception et attendre la validation** (règle 3 de `Claude.md`).
- Ne jamais deviner l'API d'une bibliothèque : vérifier la version installée et la doc.

## 5. Le dépôt est PUBLIC

- Aucun secret : `.env` est ignoré, `.env.example` ne contient que des valeurs factices.
- **Aucune donnée Semantic Scholar réelle dans le dépôt** (licence : pas de redistribution). Le fichier `tests/fixtures/semantic_scholar_batch.json` est un échantillon **fictif** écrit à la main. Ne jamais le remplacer par une vraie réponse. Ne jamais commiter d'export de la base (`data/` et `exports/` sont ignorés).
- Les fixtures arXiv (CC0) et OpenAlex sont de vraies réponses, sans souci de licence.
- Le README doit continuer à citer arXiv, OpenAlex et Semantic Scholar.

## 6. Commandes

```
uv sync
cp .env.example .env                       # puis adapter les mots de passe si besoin
docker compose up -d --build --wait        # Neo4j, PostgreSQL, API
uv run pytest                              # unitaires (sans réseau)
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run python -m graphrag.cli.ingest [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--limit N]
```

Tests d'intégration : charger d'abord `NEO4J_PASSWORD` et `POSTGRES_DSN` dans l'environnement (valeurs dans `.env`), puis `uv run pytest -m integration`. Ils n'effacent que leurs propres données (ids `9999.*`, clés `ATEST*`), pas le corpus.

Corpus complet : `uv run python -m graphrag.cli.ingest` (fenêtre par défaut = 365 jours). Rejouable sans doublon. Demander l'accord avant de le lancer (plusieurs minutes, remplit la base).

## 7. Code : où est quoi

- `src/graphrag/domain/` : modèles (`Article`, `Author`, `WorkRecord`, `ReferenceRecord`, `GraphCounts`) et identifiants arXiv/OpenAlex. Python pur.
- `src/graphrag/ports/` : `PaperSource`, `AuthorSource`, `CitationSource`, `GraphStore`, `HealthProbe`.
- `src/graphrag/adapters/` : `arxiv.py` (limite 3 s entre requêtes), `openalex.py`, `semantic_scholar.py`, `neo4j_store.py` (Cypher à la main), `http.py` (retry, limiteur), `health.py`.
- `src/graphrag/cli/ingest.py` : orchestration (articles page par page, puis auteurs, puis citations).
- `src/graphrag/api/app.py` : `GET /health` seulement pour l'instant.
- `tests/unit` (sans réseau), `tests/integration` (services Docker), `tests/fixtures`.

## 8. Pièges rencontrés

- **PostgreSQL natif Windows sur le port 5432** : le conteneur est exposé sur **5433** (`POSTGRES_DSN` dans `.env.example`).
- **Docker Desktop** doit être lancé (`C:\Users\amaur\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe`) ; le démon met environ une minute à démarrer.
- **Semantic Scholar anonyme** : le pool partagé répond souvent `429`. Fenêtres ouvertes courtes : retry fréquent (toutes les 10 s, 60 essais, environ 10 min par lot), sonde de nuit qui relance `--citations-only` au premier 200. Une clé gratuite est possible mais non nécessaire.
- **API arXiv** : renvoie parfois une entrée d'erreur dans une réponse HTTP 200 (traitée comme erreur à réessayer). Respecter 1 requête / 3 s. `max_results=0` provoque une erreur : ne pas l'utiliser.
- **PowerShell 5.1** : `Set-Content -Encoding utf8` ajoute un BOM qui casse `pyproject.toml` ; écrire en UTF-8 sans BOM. `Out-String` retarde l'affichage ; pour suivre un traitement long, lancer en arrière-plan sans `Out-String`.
- **Bash (Git Bash)** : les chemins `/tmp` ne sont pas ceux de Python Windows ; utiliser le dossier scratchpad. Les gros blocs de heredoc se sont mal comportés une fois : préférer l'outil d'écriture de fichiers.
- Avertissements `LF will be replaced by CRLF` de git : sans conséquence.
- L'avertissement Starlette sur `httpx` (« install `httpx2` ») vient du client de test ; sans impact pour l'instant.
- L'API GitHub non authentifiée donne une 404 si le dépôt est privé ; il est public maintenant, donc l'état de la CI se lit via `https://api.github.com/repos/amaury2504/GraphRAG/actions/runs`.

## 9. Suites logiques

1. Demander à l'utilisateur s'il veut l'**ingestion du corpus complet**, et vérifier à la main les deux liens de citation.
2. Obtenir son feu vert pour l'**étape 3** (découpage du résumé, embeddings, index vectoriel et texte, recherche hybride, `/ask` en `vector` et `hybrid`) : d'abord présenter la conception de la recherche hybride, proposer deux options pour les décisions ouvertes du § 3, vérifier l'API Neo4j (index vectoriel/texte) dans la doc de la version retenue.
3. Penser à : choix d'une licence, mise à jour de la section « Commandes » de `Claude.md` à chaque nouvelle commande.
