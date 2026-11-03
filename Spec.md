# SPEC — V1 : GraphRAG de veille scientifique (IA et RAG)

Spécification de la V1, en ligne droite. Les règles de travail et d'architecture sont dans `CLAUDE.md`. Les choix marqués « proposition » sont modifiables avec l'accord de l'utilisateur ; tout ce qui est dans « Décisions ouvertes » ne doit pas être tranché sans lui.

## 1. Objectifs

1. Faire progresser l'auteur sur : RAG, embeddings, recherche vectorielle et hybride, base de graphe, PostgreSQL, Docker, CI/CD, évaluation d'un RAG, suivi d'usage.
2. Obtenir un premier chiffre de qualité sur un jeu de test, en comparant trois configurations : `vector`, `hybrid`, `graph`.
3. Permettre à d'autres personnes d'utiliser l'application, avec un suivi mesuré de cet usage.

Le cœur du système (schéma du graphe, élargissement par le graphe, classement) est écrit par l'auteur avec Claude Code, sous les règles de `CLAUDE.md`. LangChain ne sert qu'aux adaptateurs.

## 2. Périmètre

| Fonction | V1 | Reporté |
| --- | --- | --- |
| Ingestion d'articles IA/RAG depuis arXiv | Oui | |
| Citations et auteurs via OpenAlex ou Semantic Scholar | Oui | |
| Découpage, embeddings, index vectoriel | Oui | |
| Recherche hybride (vecteurs + mots-clés) | Oui | |
| Élargissement par le graphe (1 saut, Cypher écrit à la main) | Oui | |
| Extraction de méthodes et de benchmarks par LLM | Petit lot de test | Tout le corpus, après mesure du coût |
| Reranking | Optionnel | |
| Réponse avec citations | Oui | |
| Jeu de test et comparaison des trois configurations | Oui | |
| Comptes simples et journal d'usage | Oui | OAuth2/OIDC |
| Routage entre modes de recherche (LangGraph) | | V2 |
| Agent qui choisit ses outils | | V2 ou après |
| Résumés par communautés (questions globales) | | Après la V2 |
| Déploiement cloud, tableaux de bord complets | | Après la V1 |

Ne rien implémenter de la colonne « Reporté ».

## 3. Flux de requête

Ordre fixe, identique pour toute question :

```
question → recherche hybride → élargissement par le graphe → classement (optionnel) → réponse avec citations
```

Configurations, qui partagent le même code et ne diffèrent que par les étapes exécutées :

| Configuration | Étapes exécutées |
| --- | --- |
| `vector` | recherche vectorielle seule, puis réponse |
| `hybrid` | recherche hybride, puis réponse |
| `graph` | recherche hybride, élargissement par le graphe, classement optionnel, réponse |

Si le corpus ne permet pas de répondre, la réponse le dit explicitement au lieu d'inventer.

## 4. Données et graphe

**Corpus V1** : titres et résumés uniquement (le texte intégral est une extension possible). Catégories arXiv cs.CL, cs.IR, cs.AI, cs.LG, filtrées par mots-clés liés au RAG, sur les 12 à 18 derniers mois.

**Nœuds**

| Nœud | Propriétés principales | Origine |
| --- | --- | --- |
| Article | identifiant arXiv, titre, résumé, date, catégories | arXiv |
| Auteur | nom | arXiv ou source de citations |
| Chunk | texte, embedding, position | découpage du résumé |
| Méthode | nom normalisé | extraction par LLM |
| Benchmark | nom normalisé | extraction par LLM |

**Relations**

```
(Article)-[:CITES]->(Article)
(Article)-[:AUTHORED_BY]->(Auteur)
(Article)-[:HAS_CHUNK]->(Chunk)
(Article)-[:USES_METHOD]->(Méthode)
(Article)-[:EVALUATED_ON]->(Benchmark)
```

**Index** : index vectoriel sur l'embedding des chunks ; index texte intégral pour les mots-clés.

**Ingestion** : rejouable sans doublons (un article déjà présent n'est pas réinséré). Les clés d'unicité et la stratégie d'écriture sont à proposer à l'utilisateur à l'étape 2.

## 5. API (FastAPI)

| Route | Rôle |
| --- | --- |
| `POST /auth/register`, `POST /auth/login` | Créer un compte, obtenir un jeton |
| `POST /ask` | Corps : question + configuration (`vector`, `hybrid`, `graph`). Réponse : texte, articles cités (identifiants arXiv), durée, tokens consommés |
| `GET /papers/{id}` | Fiche d'un article et ses voisins dans le graphe |
| `GET /health` | État de l'API et des bases |

L'ingestion et l'évaluation sont des commandes (`cli/`), pas des routes.

Exigences :

1. Chaque réponse liste les articles utilisés.
2. Les trois configurations exécutent le même flux, à une étape près.
3. Chaque requête enregistre utilisateur, configuration, durée, tokens.
4. Les clés des modèles viennent des variables d'environnement.
5. Les seuils de latence et de coût par requête sont fixés après la première mesure, pas avant.

## 6. Évaluation

- Jeu de test : une cinquantaine de questions réparties sur quatre types (factuelle, relationnelle, comparative, évolutive). Générées par un LLM à partir d'articles du corpus, validées à la main : question, réponse attendue, articles sources attendus. Fichier versionné dans le dépôt.
- Le jeu de test est construit **avant** le graphe (étape 4).
- Métriques : rappel@k des articles attendus, MRR, fidélité et exactitude (LLM juge, contrôlé à la main sur un échantillon), durée et tokens par requête.
- Résultats rapportés par type de question et par configuration.
- Limites à respecter dans le rapport : avec une cinquantaine de questions, de petits écarts ne sont pas significatifs ; ne pas présenter le graphe comme meilleur sans preuve par type de question.

## 7. Multi-utilisateurs et suivi d'usage

- PostgreSQL : comptes et journal d'usage. Neo4j : graphe, vecteurs, texte.
- Inscription et connexion simples avec jeton. Limite de requêtes par utilisateur et par jour.
- Journal par requête : utilisateur, horodatage, configuration, durée, tokens, erreur éventuelle, retour de l'utilisateur (utile / pas utile).
- Commande de rapport : utilisateurs actifs, requêtes par jour, coût total, latence médiane, taux d'erreur, part de retours positifs.
- Données personnelles : le système stocke des e-mails et des questions. Prévoir l'information des utilisateurs et la suppression d'un compte. La conformité RGPD est à vérifier auprès d'une source officielle avant ouverture au public.

## 8. Qualité et exécution

- `docker compose up` démarre l'API, Neo4j et PostgreSQL ; configuration par variables d'environnement avec un fichier d'exemple.
- CI à chaque envoi : lint, types, tests unitaires et d'intégration, construction de l'image Docker de l'API.
- Tests : logique métier sans réseau ; adaptateurs avec faux LLM et base de test ; un test de bout en bout sur un petit corpus fixe (question → réponse avec au moins une citation attendue).
- Hors V1 : déploiement cloud, déploiement continu, Prometheus/Grafana.

## 9. Étapes et critères d'acceptation

Dans l'ordre. S'arrêter à la fin de chaque étape (règle 1 de `CLAUDE.md`).

1. **Socle.** Dépôt, structure de `CLAUDE.md`, Docker Compose, CI. *Accepté quand* `docker compose up` démarre les trois services et que la CI est verte.
2. **Ingestion.** Articles arXiv, auteurs et citations dans Neo4j. *Accepté quand* une seconde exécution n'ajoute aucun doublon.
3. **RAG de base.** Découpage, embeddings, recherche hybride, `/ask` en `vector` et `hybrid`. *Accepté quand* une question test renvoie une réponse avec au moins un article cité correct.
4. **Jeu de test et script d'évaluation.** *Accepté quand* le script sort rappel@k et MRR pour `vector` et `hybrid`.
5. **Graphe structurel.** Citations et auteurs, requêtes Cypher écrites à la main. *Accepté quand* les voisins renvoyés sont corrects sur une dizaine d'articles vérifiés à la main.
6. **Extraction de méthodes et de benchmarks** sur un petit lot. *Accepté quand* la précision est vérifiée à la main sur un échantillon et le coût mesuré.
7. **Configuration `graph`.** *Accepté quand* les trois configurations sont évaluées et le rapport de comparaison écrit.
8. **Comptes et journal d'usage.** *Accepté quand* le rapport d'usage se génère à partir de vraies requêtes.
9. **Ouverture à quelques testeurs.** *Accepté quand* des personnes extérieures ont posé des questions et que leurs retours sont collectés.

Livrables : dépôt, `.env.example`, jeu de test, rapport de comparaison des trois configurations, rapport d'usage.

## 10. Décisions ouvertes

À proposer à l'utilisateur, jamais à trancher seul :

- Version de Python et outils de lint, types et tests (défauts proposés dans `CLAUDE.md`).
- Volume réel du corpus (à compter avant l'ingestion) et budget associé.
- Source de citations : OpenAlex ou Semantic Scholar ; conditions d'usage et limites de débit de chaque API et d'arXiv.
- Prise en charge des index vectoriels et du texte intégral dans la version de Neo4j retenue.
- Modèle d'embeddings (et de reranking) : à départager sur le jeu de test.
- Fournisseur de LLM et budget d'appels.
- Règle de normalisation des noms de méthodes et de benchmarks, avant l'étape 6.
- Taille du petit lot d'extraction, nombre de testeurs visé, seuils de latence et de coût.

## 11. Risques

| Risque | Réponse prévue |
| --- | --- |
| Corpus trop volumineux | Compter avant d'ingérer ; réduire la période ou les mots-clés |
| Extraction de méthodes bruyante | Schéma figé, normalisation, contrôle manuel d'un échantillon, petit lot |
| API de citations limitée | Vérifier les conditions avant l'étape 2 ; accepter une couverture partielle et la mesurer |
| Le graphe n'améliore pas les résultats | Publier le résultat tel quel ; la V2 reste utile pour le routage |
| Coût des appels aux modèles | Limite par utilisateur, journal des tokens, budget fixé avant l'ouverture |
| Bibliothèques qui changent vite | Versions figées, adaptateurs derrière des interfaces |