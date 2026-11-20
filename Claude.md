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
- **Gates** : `uv run python scripts/gates.py --skip mutation --skip gitleaks` (lint, format, types, tests avec couverture, anti-contournement). C'est la commande de la CI ; elle doit être verte avant un commit. Seuil de couverture : `fail_under` dans `pyproject.toml` (85 %, mesuré à 88 %), à relever par paliers, jamais à baisser. Pas de `typing.cast` ni de contournement du typage (la gate anti-contournement le refuse).
- **PR** : une PR qui ne touche que la doc ou la roadmap se merge sans attendre la CI ; une PR contenant du code attend une CI verte (jobs `quality` et `integration`).

## Commandes

À remplir à l'étape 1, puis à maintenir à jour :

```
# installer                  : uv sync   (Python 3.14, versions figées dans uv.lock)
# démarrer les services      : cp .env.example .env && docker compose up -d --build --wait
# lancer les tests           : uv run pytest                (unitaires, sans réseau)
#                              uv run pytest -m integration (nécessite les services ; charger .env)
#                              uv run pytest -m model       (vrai modèle d'embeddings, téléchargé au 1er lancement)
# lint et types              : uv run ruff check . && uv run ruff format --check . && uv run mypy
# gates (comme la CI)        : uv run python scripts/gates.py --skip mutation --skip gitleaks
# ingestion                  : uv run python -m graphrag.cli.ingest [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--limit N]
#                              [--citations-only] [--recheck-citations]  (exit 3 = citation batches skipped, rerun later)
# indexation (morceaux)      : uv run python -m graphrag.cli.index [--limit N] [--batch-size N] [--rollback]
#                              (titre + morceau → bge-small → Neo4j ; ne traite que les articles dont le hash
#                              titre + résumé + version du chunker diffère de celui indexé : nouveaux, modifiés,
#                              ou chunker changé (CHUNKER_VERSION).
#                              Les morceaux sont immuables (clé = article#position#empreinte du texte) et jamais
#                              supprimés ; la recherche ne lit que la version courante, un instantané du corpus
#                              publié en fin de run (nœuds Version, pointeur Current, bascule en une transaction).
#                              Statuts d'une version : pending (manifeste en construction) → indexed (index vectoriel
#                              et texte intégral en ligne et couvrant tous ses morceaux, sinon exit 4 et la version
#                              active ne change pas) → active (pointée par Current, seule lisible).
#                              --rollback : revient à la version indexed précédente, sans indexer.
#                              --gc [--keep N] [--min-age-days D] [--apply] : ramasse-miettes (mark and sweep),
#                              sans indexer ni charger le modèle. Simulation par défaut ; --apply supprime les
#                              versions au-delà des N plus récentes (défaut 3, minimum 2, sauf celles de moins de
#                              D jours) puis les morceaux hors de toute version gardée et hors des morceaux que
#                              chaque article prépare pour la prochaine version (donc sans risque pendant un `index`).
#                              Code retour 2 si aucune version n'est publiée ou si --keep < 2.
#                              Supprime aussi l'index vectoriel (et les vecteurs) d'un modèle d'embedding que
#                              seules les versions abandonnées utilisaient ; jamais celui d'un ré-encodage en cours.)
#                              --reembed : bascule vers le modèle de EMBEDDING_MODEL (clé de EMBEDDING_MODELS dans
#                              adapters/sentence_transformer.py ; défaut bge-small). Chaque modèle a sa propriété
#                              (`embedding_<empreinte>`) et son index vectoriel sur les morceaux ; l'ancien reste
#                              servi pendant le calcul des nouveaux vecteurs (reprenable), puis une version lue avec
#                              le nouveau modèle est publiée (pending → indexed → active, même bascule atomique) ;
#                              `--rollback` redonne l'ancien modèle. Chaque version porte l'empreinte du modèle ;
#                              `index`, l'API, `evaluate` et `pool` refusent un modèle autre que celui de la version
#                              active. Code retour 2 sans version publiée, 4 si les index ne couvrent pas la version.
#                              Relancer `ingest` détecte les articles modifiés, puis `index` les ré-indexe.)
# LLM de /ask               : LLM_PROVIDER=mistral (défaut) | claude (commande locale `claude -p`, usage personnel, pas dans Docker)
# question (API)             : POST /ask {"question": "...", "configuration": "vector"|"hybrid", "k": 5}  (réponse : + timings)
#                              (nécessite MISTRAL_API_KEY ; modèle chargé au 1er appel ; doc interactive sur /docs)
# questions candidates       : uv run python -m graphrag.cli.generate_questions [--per-type 2] [--seed 42] [--style named|descriptive] [--dry-run]
#                              (descriptive : une question qui reprend un mot rare du titre est réécrite une fois, sinon rejetée)
#                              (appelle Mistral : 8 questions max sans --confirm-cost ; écrit eval/candidates.jsonl,
#                              à relire puis à copier dans eval/questions.jsonl avec "validated": true)
# évaluation (recherche)     : uv run python -m graphrag.cli.evaluate [--ks 1 3 5 10] [--include-unvalidated] [--split tune|report] [--with-keyword]
#                              (--split : régler sur « tune », rapporter sur « report » ; nDCG@10 et écart minimal détectable affichés)
#                              (rappel@k et MRR pour vector et hybrid, sans LLM ; questions dans eval/questions.jsonl,
#                              seules celles avec "validated": true sont utilisées ; rapport JSON dans eval/results/)
# pooling des jugements      : uv run python -m graphrag.cli.pool [--depth 10]   (sans LLM ; écrit eval/pool_review.jsonl :
#                              mettre "relevant": true/false sur chaque article, puis
#                              uv run python -m graphrag.cli.pool --apply eval/pool_review.jsonl)
# évaluation (réponses)      : (à définir)
# rapport d'usage            : (à définir)
```

## Langue

Code, identifiants et messages de commit en anglais. Documentation et échanges avec l'utilisateur en français.
