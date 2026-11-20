# GraphRAG de veille scientifique (IA et RAG)

Application qui répond à des questions sur des articles arXiv d'IA et de RAG, avec citations
(recherche hybride, élargissement par le graphe, réponse). Projet d'apprentissage, V1 en cours.
Règles du projet : `CLAUDE.md`. Spécification : `SPEC.md`.

## Démarrage

```
uv sync
cp .env.example .env        # puis changer les mots de passe
docker compose up -d --build --wait
uv run python -m graphrag.cli.ingest --limit 100
```

## Tests d'intégration

Ils vident la base : ils tournent sur un Neo4j jetable (`neo4j-test`, port 7688, données en
mémoire), jamais sur la base réelle (un test refuse une base qui contient de vrais articles).

```
docker compose --profile test up -d --wait neo4j-test
set -a; . ./.env; set +a    # NEO4J_PASSWORD, POSTGRES_DSN
NEO4J_URI=bolt://localhost:7688 uv run pytest -m integration tests/integration/test_chunk_store.py
docker compose --profile test down neo4j-test    # facultatif : la base repart vide à chaque redémarrage
```

En local, ne lancer que les fichiers d'intégration concernés par le changement (la suite complète
dure environ 35 minutes) ; la CI lance le reste.

## Données et licences

Ce dépôt contient uniquement du code et de petits échantillons de test fictifs ou libres.
La base de données (volumes Docker) n'est jamais versionnée et ne doit pas être publiée.

- Métadonnées d'articles : [arXiv](https://arxiv.org) (CC0). Ce projet n'est ni soutenu ni
  approuvé par arXiv.
- Auteurs : [OpenAlex](https://openalex.org).
- Références (citations) : [Semantic Scholar](https://www.semanticscholar.org) Academic Graph API,
  Allen Institute for AI. Les données restent soumises à leur licence ; ne pas les redistribuer.
