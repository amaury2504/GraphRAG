# GraphRAG de veille scientifique (IA et RAG)

Application qui répond à des questions sur des articles arXiv d'IA et de RAG, avec citations
(recherche hybride, élargissement par le graphe, réponse). Projet d'apprentissage, V1 en cours.
Règles du projet : `Claude.md`. Spécification : `Spec.md`.

## Démarrage

```
uv sync
cp .env.example .env        # puis changer les mots de passe
docker compose up -d --build --wait
uv run python -m graphrag.cli.ingest --limit 100
```

## Données et licences

Ce dépôt contient uniquement du code et de petits échantillons de test fictifs ou libres.
La base de données (volumes Docker) n'est jamais versionnée et ne doit pas être publiée.

- Métadonnées d'articles : [arXiv](https://arxiv.org) (CC0). Ce projet n'est ni soutenu ni
  approuvé par arXiv.
- Auteurs : [OpenAlex](https://openalex.org).
- Références (citations) : [Semantic Scholar](https://www.semanticscholar.org) Academic Graph API,
  Allen Institute for AI. Les données restent soumises à leur licence ; ne pas les redistribuer.
