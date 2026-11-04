# GraphRAG de veille scientifique (IA et RAG) — V1

Règles permanentes du projet. La spécification complète est dans `SPEC.md` : la lire en entier avant toute tâche, puis relire la section de l'étape en cours.

## Le projet en une phrase

Application qui répond à des questions sur des articles arXiv d'IA et de RAG, avec un flux en ligne droite (recherche hybride → élargissement par le graphe → classement optionnel → réponse avec citations), plus une évaluation qui compare trois configurations (`vector`, `hybrid`, `graph`). La V2 (hors périmètre ici) ajoutera LangGraph.

## Règles de travail

1. **Une étape à la fois.** Les étapes sont numérotées dans `SPEC.md`. À la fin d'une étape : lancer les tests, comparer le résultat au critère d'acceptation de l'étape, résumer en quelques lignes ce qui est fait et ce qui manque, puis **s'arrêter et attendre le feu vert** avant l'étape suivante.
2. **Ne pas trancher seul les « Décisions ouvertes »** de `SPEC.md`. Proposer deux options avec leurs avantages et inconvénients, puis demander.
3. **Expliquer avant d'implémenter** ces trois parties, que l'utilisateur veut maîtriser : la recherche hybride, le schéma du graphe avec les requêtes d'élargissement, le jeu de test avec ses métriques. Présenter la conception en quelques lignes, attendre la validation, puis coder.
4. **Ne jamais deviner l'API d'une bibliothèque.** Vérifier la version installée et la documentation officielle. En cas de doute, le dire plutôt que d'inventer un nom de fonction ou de paramètre.
5. **Coûts.** Tout appel à un vrai LLM ou à un service payant hors des évaluations demandées exige un accord préalable. L'extraction par LLM se teste d'abord sur un petit lot dont la taille est fixée par l'utilisateur.
6. **Secrets.** Aucune clé dans le dépôt ni dans les logs. Les variables d'environnement sont décrites dans `.env.example`.
7. **Petits commits**, un sujet par commit.

## Architecture (règles dures)

```
src/graphrag/   # paquet Python ; les dossiers ci-dessous sont ses sous-paquets
  domain/      # Article, Auteur, Méthode, Benchmark, schéma du graphe
  ports/       # interfaces : Retriever, GraphStore, LLM, Embedder
  adapters/    # implémentations : Neo4j, HuggingFace, fournisseur de LLM, LangChain
  pipeline/    # le flux en ligne droite
  api/         # FastAPI
  cli/         # ingestion, évaluation, rapport d'usage
tests/
```

- `domain/` n'importe ni LangChain, ni Neo4j, ni FastAPI, ni aucune bibliothèque d'IA.
- LangChain n'est utilisé que dans `adapters/`. LangGraph n'est pas utilisé en V1.
- `pipeline/` de la V1 est une suite d'appels dans un ordre fixe : pas de routage, pas de boucle, pas de choix d'outil par le modèle.
- Les requêtes Cypher d'élargissement par le graphe sont écrites à la main. Ne pas utiliser de chaîne qui génère le Cypher à partir d'une question.
- Les conversions vers les types de LangChain (par exemple `Document`) se font aux frontières, dans les adaptateurs.

## Qualité

- Typage des signatures publiques, lint, formatage et tests exécutés avant de déclarer une étape terminée.
- Outils proposés par défaut, modifiables avec l'accord de l'utilisateur : `pytest`, `ruff`, `mypy`.
- Tests unitaires et d'intégration sans réseau et sans vrai LLM : utiliser un faux LLM et une base de test.
- Les tests d'évaluation (jeu de test d'une cinquantaine de questions) appellent de vrais modèles : ils se lancent à la demande, jamais automatiquement.
- Versions des dépendances et des images Docker figées.

## Commandes

À remplir à l'étape 1, puis à maintenir à jour :

```
# installer                  : uv sync   (Python 3.14, versions figées dans uv.lock)
# démarrer les services      : cp .env.example .env && docker compose up -d --build --wait
# lancer les tests           : uv run pytest                (unitaires, sans réseau)
#                              uv run pytest -m integration (nécessite les services ; charger .env)
#                              uv run pytest -m model       (vrai modèle d'embeddings, téléchargé au 1er lancement)
# lint et types              : uv run ruff check . && uv run ruff format --check . && uv run mypy
# ingestion                  : uv run python -m graphrag.cli.ingest [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--limit N]
#                              [--citations-only] [--recheck-citations]  (exit 3 = citation batches skipped, rerun later)
# indexation (morceaux)      : uv run python -m graphrag.cli.index [--limit N] [--batch-size N]
#                              (titre + morceau → bge-small → Neo4j ; ne traite que les articles sans morceaux)
# question (API)             : POST /ask {"question": "...", "mode": "vector"|"hybrid", "k": 5}
#                              (nécessite MISTRAL_API_KEY ; modèle chargé au 1er appel ; doc interactive sur /docs)
# évaluation                 : (à définir)
# rapport d'usage            : (à définir)
```

## Langue

Code, identifiants et messages de commit en anglais. Documentation et échanges avec l'utilisateur en français.