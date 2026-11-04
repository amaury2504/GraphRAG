# Brief de reprise (pour la prochaine session Claude)

À lire après `Claude.md` et `Spec.md`. État au 2026-11-04 (dernier commit : `d8dcd39`, poussé, CI verte). Ce fichier résume ce qui est fait, ce qui a été décidé, ce qui reste à améliorer, et les pièges déjà rencontrés, pour ne pas les redécouvrir.

## 1. Où on en est

| Étape | État |
| --- | --- |
| 1. Socle (structure, Docker Compose, CI) | **Terminée**, CI verte. |
| 2. Ingestion (arXiv, auteurs, citations → Neo4j) | **Terminée** : corpus complet ingéré le 2026-11-04. |
| 3. RAG de base | **Terminée** : découpage, embeddings bge-small, corpus indexé (3 879 morceaux), recherche `vector` et `hybrid` (RRF), réponse Mistral avec citations vérifiées, `POST /ask`. Un essai réel a réussi le 2026-11-04 (1 351 tokens, 4 articles cités, tous pertinents). L'utilisateur est passé à l'étape 4 sans contester le critère d'acceptation. |
| 4. Jeu de test et script d'évaluation | **En cours.** Le script d'évaluation est écrit et testé ; le jeu de test n'a que **3 questions validées sur une cinquantaine visées**. Le critère d'acceptation (« le script sort rappel@k et MRR pour `vector` et `hybrid` ») n'est pas atteint tant qu'il n'y a pas un jeu de test validé de taille raisonnable. |
| 5 à 9 | Non commencées. |

**État de la base locale** (2026-11-04) :
- 2 827 articles, 10 383 auteurs, 11 203 `AUTHORED_BY`, 924 `CITES` (539 articles citent, 526 sont cités).
- **593 articles n'ont aucun auteur** (inconnus d'OpenAlex).
- 3 879 `Chunk` : 1 052 articles ont 2 morceaux, les autres 1. Index vectoriel `chunk_embedding` (384, cosinus) et texte intégral `chunk_text` (analyseur par défaut).
- Semantic Scholar : 2 324 articles trouvés, 3 inconnus, 93 sans référence arXiv.

**Tests** : 369 unitaires, 30 d'intégration (vrai Neo4j), 1 test sur le vrai modèle (`-m model`). `ruff` et `mypy --strict` passent. Chaque brique a été vérifiée par tests de mutation (casser volontairement le code et constater l'échec des tests).

**Consommation Mistral cumulée** : environ 16 600 tokens (essais, une vraie question, deux lots de génération de questions). Aucun coût attendu sur l'offre gratuite.

### État précis de l'étape 4
- Écrit : `domain/evaluation.py` (questions, rappel@k, MRR, intervalle bootstrap), `pipeline/evaluation.py` (évaluation par configuration, différences appariées), `cli/evaluate.py` (aucun LLM ; tableaux par type de question ; **jamais d'étoile « significatif » sous 10 questions** ; rapport JSON dans `eval/results/`, ignoré par git), `adapters/question_file.py` (JSON Lines), `pipeline/question_generation.py` et `cli/generate_questions.py` (génération des candidats par Mistral).
- `eval/questions.jsonl` : 3 questions validées par l'utilisateur : `factual-01` (ARIA, `2610.07902`), `factual-02` (savoirs indonésiens, `2601.12921`), `comparative-01` (`2602.02559` et `2605.03989` ; la réponse attendue précise que les deux métriques ne se comparent pas).
- La génération respecte : groupes d'articles choisis **par proximité de thème** (voisins vectoriels et mots-clés alternés, jamais par le graphe) ; les articles des questions déjà validées (et leurs doublons de titre) ne sont jamais réutilisés ; consigne qui exige de nommer les sujets ; contrôle automatique des formulations vagues (« these works », « the two approaches »…) avec une réécriture ; le modèle peut répondre `{"skip": ...}` ; **8 questions maximum sans `--confirm-cost`**.
- **Les 10 prochaines questions (5 factuelles, 5 comparatives) sont choisies en simulation seulement** (`--dry-run`, graines 42 et 43) : rien n'a été généré ni écrit, aucun appel Mistral. **Décision en attente de l'utilisateur** : faire d'abord le point 1 des améliorations (style descriptif et détecteur de fuite) puis générer, ou générer tout de suite avec la consigne actuelle (voir la fin du § 4).
- Les chiffres de rappel et de MRR obtenus pendant la session portaient sur des candidats non validés : **ils ne valent rien** et ne doivent pas être cités.

## 2. Décisions prises (ne pas les rouvrir sans raison)

- **Python 3.14.2**, `uv` (versions figées dans `uv.lock`), `pytest`, `ruff`, `mypy --strict`.
- **Corpus « A »** : catégories cs.CL/IR/AI/LG, résumé contenant « retrieval-augmented » (ou « retrieval augmented »), 12 mois. 2 829 articles comptés le 2026-11-03, 2 827 ingérés. Titres et résumés seulement (pas les PDF).
- **Auteurs : OpenAlex**, sans clé, par DOI arXiv, lots de 100. Un article inconnu d'OpenAlex n'a pas de nœud `Author`.
- **Citations : Semantic Scholar**, sans clé, lots de 100, `CITES` seulement entre articles du corpus. Reprise par le fichier local `data/citations_checked.txt` (ignoré par git). Une réponse qui porte un autre numéro arXiv que celui demandé est un **alias** (même travail déposé deux fois) : ses références sont gardées sous le numéro demandé, sauf si ce numéro est celui d'un autre article du même lot (vrai décalage, erreur).
- **Clés d'unicité** : `Article.arxiv_id` (sans version), `Author.key` (id OpenAlex), `Chunk.key` (`arxiv_id#position`). Écriture par `MERGE`. Un article déjà présent n'est jamais modifié.
- **Embeddings : `BAAI/bge-small-en-v1.5`**, local, processeur, 384 dimensions, 512 tokens, **révision figée** (`5c38ec7…`). L'instruction de requête de la fiche du modèle est ajoutée aux questions, jamais aux passages. PyTorch en version processeur sous Linux (CI, Docker), via l'index PyTorch.
- **Morceaux** : 200 mots au plus, coupés sur les phrases, une phrase de chevauchement. **Texte indexé = « Titre. Morceau »** (option A) ; le texte seul du morceau est aussi conservé.
- **Recherche** : 20 candidats par recherche (par le sens, par mots-clés), **fusion RRF `1/(60 + rang)` en Python** dans `pipeline/`, un article au plus par classement, meilleur morceau gardé. Les questions des utilisateurs sont transformées en requête Lucene avec tous les opérateurs échappés. L'option B (somme pondérée de scores normalisés) est à essayer après l'étape 4.
- **LLM : Mistral par l'API**, `mistral-small-latest` (réglable par `MISTRAL_MODEL`), appelé directement en `httpx` (pas de LangChain pour l'instant). Limites vues : 100 requêtes et 100 000 tokens par minute. Température 0, 700 tokens de réponse au plus. Claude Haiku 5.5 reste l'option de secours. Pas de Claude « headless » : il consommerait l'abonnement, serait lourd, et ne convient pas à une application.
- **Budget** : compteur de tokens qui **alerte une fois** au-delà de `TOKEN_ALERT` (500 000 par défaut) sans jamais bloquer. Il vit en mémoire ; le vrai rapport d'usage est prévu à l'étape 8.
- **Réponse** : sources numérotées dans la consigne, les `[n]` du modèle deviennent `[arXiv:id]`, tout numéro hors liste est supprimé, aucun appel LLM s'il n'y a aucune source.
- **API** : `POST /ask {question, mode: "vector" | "hybrid", k}` ; le modèle d'embeddings et le client Mistral se chargent à la première question ; 503 sans clé, 502 si Mistral échoue ; aucun détail d'erreur n'est exposé.
- **Jeu de test (décidé le 2026-11-04)** : articles groupés par proximité de thème (jamais par le graphe), questions générées par Mistral puis validées à la main. **L'étape 4 ne contient que des questions factuelles et comparatives.** Les noms des systèmes cités dans les questions facilitent la recherche : à signaler dans le rapport.
- **Reporté aux étapes 5 à 7** (à trancher alors) :
  - Questions **relationnelles et évolutives** : avec des articles seulement voisins par le thème, le modèle invente des liens ou des filiations (premier lot : relational-02 chronologiquement impossible, deux « évolutions » inventées). Les reconstruire quand le graphe donne un sens à « relation ».
  - Questions fondées sur les **citations** : elles favorisent le graphe par construction, et les citations viennent de Semantic Scholar, **dont les données ne doivent pas aller dans le dépôt public**. Les auteurs viennent d'OpenAlex (CC0), sans cette restriction.
  - Questions **temporelles** (« parus avant mars 2026 ») : demandent un filtre par date, absent de la recherche.
  - **Dates arXiv** : 94 articles sur 2 827 ont une date de v1 antérieure de 2 mois ou plus au mois de leur identifiant (par exemple M-RAG `2603.26667`, v1 au 6 janvier 2026). Notre base reproduit fidèlement l'API arXiv ; l'écart vient de la source. À garder en tête pour toute logique fondée sur les dates.
  - Les réponses attendues séparent ce que dit l'article de ce qui est déduit (déjà demandé dans la consigne de génération).

## 3. Décisions encore ouvertes (à proposer en deux options, jamais à trancher seul)

- **Prochaine action immédiate** : point 1 des améliorations d'abord, ou génération des 10 questions tout de suite (§ 4, fin).
- **Version de Neo4j** : 5.26.32 LTS en place ; index vectoriel et texte intégral vérifiés dessus. Alternative : 2026.10 (`db.index.vector.queryNodes` y est déprécié depuis 2026.04).
- **Densité du graphe de citations** (à l'étape de l'élargissement par le graphe) : 924 `CITES` pour 2 827 articles. Option A : garder les articles cités hors corpus comme nœuds minces (id + titre, sans résumé). Option B : élargir la fenêtre (par exemple 24 mois).
- **Doublons arXiv** : un même travail déposé deux fois donne deux nœuds `Article`, par exemple `2608.12129` et `2606.15971`. Pour l'évaluation, un titre identique est traité comme équivalent (`equivalents` dans les questions). Reste à décider de leur traitement dans la recherche (une relation `SAME_AS` est proposée au § 4, point 11).
- **Questions en français** : traduire la question avant la recherche, ou prendre un modèle d'embeddings multilingue (§ 4, point 7).
- **Juge LLM des réponses** : quel modèle, différent du générateur (§ 4, point 16).
- Fusion hybride option B, reclasseur, normalisation des noms de méthodes et de benchmarks (avant l'étape 6), taille du lot d'extraction, nombre de testeurs, seuils de latence et de coût.
- Licence du dépôt (aucun fichier `LICENSE` pour l'instant).

## 4. Améliorations à faire (texte repris tel quel de l'analyse du 2026-11-04)

Voici le point, détaillé jusqu'à l'implémentation, et appuyé sur ce que j'ai mesuré dans cette session. J'ai mis la génération des 10 questions en pause, parce que le premier point modifie la façon de les écrire.

Trois mesures faites à l'instant :
- **593 articles sur 2 827 (21 %) n'ont aucun auteur.**
- Un seul couple de titres identiques existe dans le corpus.
- L'analyseur de texte `english` est disponible dans notre Neo4j.

---

### P0 : fiabilité de la mesure (avant de générer plus de questions)

**1. Les noms des systèmes dans les questions rendent la recherche trop facile.**
- **Le problème.** Les questions citent ARIA, GeoEvolver… Or on indexe « Titre. Morceau » (option A), donc le nom tombe dans l'index texte dès le rang 1. D'où un MRR d'environ 0,9 sur les candidats, un chiffre flatteur. De plus, le LLM reformule le résumé indexé, ce qui avantage aussi la recherche par le sens.
- **La correction.**
  - Ajouter un champ `style: "named" | "descriptive"` dans `eval/questions.jsonl`. Une question descriptive explique le problème sans nommer le système, comme le ferait quelqu'un qui ne connaît pas l'article.
  - Ajouter un **détecteur de fuite** sur le même modèle que `find_vague_wording`. Il calcule la fréquence documentaire de chaque mot des 2 827 titres, et rejette (avec une réécriture) toute question descriptive contenant un mot rare du titre (fréquence ≤ 3, par exemple « ARIA », « TRATE »).
  - Enregistrer pour chaque question le recouvrement de mots entre question et titre, afin de rapporter les scores par style et par niveau de recouvrement.
  - Faire écrire environ 10 questions **par vous**, sans LLM, pour casser le biais des questions synthétiques.

**2. La vérité terrain est incomplète.**
- **Le problème.** Le corpus ne parle que de RAG : d'autres articles que l'article attendu peuvent répondre, et ils comptent alors comme des échecs.
- **La correction.**
  - Faire un **pooling** : pour chaque question, rassembler les 10 premiers résultats de `vector` et de `hybrid` qui ne sont pas attendus, et vous les faire juger. Ils sont stockés dans un champ `also_relevant`, ce qui permet d'ajouter un nDCG@10 gradué.
  - Point méthodologique important : il faudra **refaire le pooling quand `graph` arrivera**. Sinon, ses bons résultats inédits compteront comme faux, et le graphe sera pénalisé injustement.
  - Ajouter environ 5 **questions sans réponse dans le corpus**, pour mesurer si le système sait dire « je ne trouve pas ».

**3. La puissance statistique est limitée.**
- **Le problème.** Avec 50 questions et environ 20 % de questions où une seule des deux configurations trouve l'article, l'écart minimal détectable sur le rappel@5 est d'environ **±0,12**. En dessous, aucune conclusion n'est possible.
- **La correction.** Le rapport doit l'afficher. Tout réglage (la constante de RRF, par exemple) se choisit sur une moitié des questions et se rapporte sur l'autre.

### P1 : qualité de la recherche et des réponses (mesurable sur le jeu de test)

**4. L'analyseur du texte intégral.**
- **Le problème.** L'analyseur par défaut, `standard-no-stop-words`, ne retire aucun mot vide et ne réduit pas les mots à leur racine. La question entière (« What are the two… ») part donc en requête OU sur tous ses mots, et « agents » ne correspond pas à « agent ».
- **La correction.**
  - Recréer `chunk_text` avec `english` (`OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}` ; la clé est à vérifier dans la documentation 5.26).
  - `IF NOT EXISTS` ne remplace pas un index déjà créé : il faut un contrôle dans `ensure_chunk_schema` (`SHOW FULLTEXT INDEXES YIELD options`), comme celui qui existe pour les dimensions, puis supprimer et recréer l'index (quelques secondes pour 3 879 nœuds).
  - Ajouter une configuration de diagnostic `keyword` (texte intégral seul) à l'évaluation. On saura ainsi ce que chaque moitié apporte à la fusion.

**5. Une constante RRF de 60 est trop plate pour des listes de 20 résultats.**
- **Le problème.** Le 1er rang vaut 1/61 ≈ 0,0164 et le 20ᵉ vaut 1/80 = 0,0125. Résultat : un article **classé 20ᵉ dans les deux listes (0,025) bat un article classé 1ᵉʳ dans une seule (0,0164)**. Avec k = 10, c'est l'inverse : 2/30 ≈ 0,067 contre 1/11 ≈ 0,091.
- **La correction.**
  - Tester k ∈ {10, 30, 60} sur la moitié de réglage des questions.
  - Porter `CANDIDATES` de 20 à 50 : le coût est négligeable, et les deux listes se recouvriront davantage.
  - Corriger un piège latent : un article a au plus 2 morceaux, donc 20 morceaux ne garantissent que 10 articles. Avec `--ks 20`, le rappel@20 serait sous-estimé. Il faut `candidates = max(50, 2k)`.

**6. Passer le résumé complet au LLM, et non un seul morceau.**
- **Le problème.** On garde le meilleur morceau par article et on l'envoie seul au LLM. Si la réponse est dans l'autre morceau, elle est perdue.
- **La correction.** Faire `VECTOR_SEARCH` et `KEYWORD_SEARCH` renvoyer aussi `a.abstract` (dans [neo4j_store.py](src/graphrag/adapters/neo4j_store.py)), et construire la consigne avec. Le contexte grandit au plus d'un facteur 2 (en moyenne 1,37 morceau par article). On cherche sur les morceaux, on rédige sur l'article entier.
- Une hypothèse à mesurer : un résumé tient presque toujours dans les 512 tokens de bge-small, donc **ne pas découper du tout** pourrait suffire et supprimer le dédoublonnage.

**7. Les questions en français.**
- **Le problème.** bge-small-en et l'analyseur `english` sont tous deux en anglais. Deux options (décision ouverte) :
  - **A. Traduire la question en anglais** avec un appel Mistral d'environ 100 tokens avant la recherche. Ça sert **les deux moitiés** de la recherche hybride, et permet aussi de reformuler la question.
  - **B. Prendre un modèle multilingue** (par exemple `intfloat/multilingual-e5-small`, 384 dimensions, à vérifier). Ça n'aide pas la recherche par mots-clés, et ça impose de tout réindexer.
- Je recommande A.

**8. Un reclasseur (« classement optionnel » de la spécification).**
- `CrossEncoder` est présent dans `sentence-transformers` 6.2 (vérifié dans le paquet installé). On reclasserait les 20 premiers résultats de la fusion.
- `cross-encoder/ms-marco-MiniLM-L-6-v2` prendrait de l'ordre de 0,1 à 0,3 seconde sur le processeur. Il a été entraîné sur des recherches web, et son adéquation à des résumés scientifiques est à mesurer.

**9. Un même nombre de dimensions ne protège pas d'un changement de modèle.**
- **Le problème.** bge-small et MiniLM ont **tous deux 384 dimensions**. Notre contrôle actuel laisserait donc se mélanger des vecteurs incompatibles sans rien signaler.
- **La correction.** Créer un nœud `(:IndexInfo {name, model, revision, dims, text_template: "title. chunk"})`, vérifié par `ensure_chunk_schema`, qui refuse toute différence.

### P2 : le graphe (étape 5), c'est là que se joue le « Graph » de GraphRAG

**10. L'élargissement doit être pondéré, puis re-noté par rapport à la question.**
- **Le principe.** Un voisin n'a aucune pertinence par lui-même. Si on l'ajoute tel quel, on ajoute du bruit. La requête écrite à la main (à valider sur 5.26 : `CALL (s) {}` et `vector.similarity.cosine`) ressemblerait à ceci :
  ```cypher
  MATCH (s:Article) WHERE s.arxiv_id IN $seeds
  CALL (s) {
    MATCH (s)-[:CITES]-(n:Article) RETURN n, 1.0 AS w
    UNION ALL
    MATCH (s)-[:AUTHORED_BY]->(p:Author)<-[:AUTHORED_BY]-(n:Article)
    RETURN n, 1.0 / log(1 + COUNT { (p)<-[:AUTHORED_BY]-() }) AS w   // Adamic-Adar
  }
  WITH n, sum(w) AS graph_score WHERE NOT n.arxiv_id IN $seeds
  MATCH (c:Chunk)-[:PART_OF]->(n)
  WITH n, graph_score, max(vector.similarity.cosine(c.embedding, $qvec)) AS query_sim
  RETURN n.arxiv_id, graph_score, query_sim ORDER BY query_sim DESC LIMIT $limit
  ```
- **Les détails qui comptent.**
  - Le terme Adamic–Adar (1/log du nombre d'articles de l'auteur) empêche les auteurs très prolifiques de relier tout le corpus.
  - Plafonner le nombre de voisins par article de départ (`LIMIT` dans la sous-requête).
  - La fusion finale devient un RRF à trois listes (sens, mots-clés, graphe trié par `query_sim`), avec un poids plus faible pour la liste du graphe.
  - Les méthodes et benchmarks de l'étape 6 s'ajouteront de la même façon, pondérés par la rareté : HotpotQA, cité partout, ne relie presque rien.

**11. La couverture du graphe est faible.**
- **Les auteurs.** 593 articles n'en ont aucun. On peut faire une seconde passe OpenAlex par titre (filtre `title.search`, à vérifier), en n'acceptant une correspondance que si au moins la moitié des noms d'auteurs arXiv concordent. Créer des auteurs à partir des seuls noms arXiv relierait des homonymes, ce qui fausserait l'élargissement.
- **Les citations.** 924 liens seulement. On peut lire aussi les citations **entrantes** de Semantic Scholar (champ `citations`), ce qui récupère un lien quand la liste de références de l'un des deux articles est incomplète.
- **La décision A/B déjà ouverte** (garder les articles cités hors corpus comme nœuds minces) devient centrale. Elle permet le **couplage bibliographique** : deux articles qui citent tous deux le RAG original de 2020 sont liés, et le graphe se densifie fortement.
- **Les doublons.** Un seul couple dans le corpus : une relation `SAME_AS` suffit, sans fusionner les nœuds.

### P3 : écarts avec la spécification et exploitation

12. **`/ask` ne renvoie pas la durée**, que `Spec.md` exige. Il faut l'ajouter par étape (recherche, LLM), et renommer `mode` en `configuration` comme dans la spécification.
13. **Le test de bout en bout sur un petit corpus fixe**, exigé par `Spec.md`, manque. Il passerait par FastAPI et le vrai Neo4j, avec un faux LLM et un faux modèle d'embeddings déterministe, et vérifierait qu'une citation attendue ressort bien.
14. **Limiter le débit vers Mistral avant d'évaluer les réponses.** 150 réponses d'environ 1 400 tokens font environ 200 000 tokens, pour une limite de 100 000 tokens par minute. Sans limiteur proactif, on aura des salves de 429, et nos 5 tentatives peuvent s'épuiser. On peut adapter `MinIntervalLimiter` aux tokens.
15. **Le conteneur de l'API.**
    - Définir `HF_HUB_OFFLINE=1` une fois le modèle en cache : sinon chaque chargement interroge Hugging Face, comme le montre l'avertissement qu'on voit à chaque fois.
    - Ajouter un préchauffage du modèle au démarrage.
    - Faire signaler par `/health` le nombre d'**articles non indexés** : l'ingestion ne lance pas `cli.index` toute seule.
16. **Le juge LLM des réponses.**
    - Utiliser un modèle différent du générateur, pour éviter qu'il favorise ses propres réponses.
    - Vérifier la fidélité **affirmation par affirmation** contre les morceaux cités.
    - Contrôler son accord avec vos jugements sur 20 réponses (coefficient kappa).

---

> **Mise à jour (2026-11-04, plus tard)** : le point 1 est implémenté (champ `style`, `--style named|descriptive`, détecteur de fuite à fréquence ≤ 3, `title_overlap` enregistré, tableaux par style dans `evaluate`). Reste : générer les 10 questions (3 factuelles + 2 comparatives descriptives, 2 factuelles + 3 comparatives nommées) après accord sur l'appel Mistral.

> **Mise à jour (2026-11-04, fin de journée)** : implémentés sans appel Mistral, un commit chacun, tous poussés : point 4 (analyseur `english` + `--with-keyword` dans `evaluate`, l'index se reconstruit seul si l'analyseur diffère), point 5 en partie (50 candidats, `max(50, 2k)` ; la constante RRF reste à régler sur le jeu de test), point 6 (le LLM reçoit le résumé complet, `SearchHit.abstract`), point 9 (`IndexInfo` : fiche modèle/révision/dimensions/modèle de texte, vérifiée par `evaluate` et l'API, enregistrée sur la base réelle), point 12 (`timings` dans `/ask`, `mode` renommé `configuration`), point 13 (test de bout en bout). **Restent** : points 2, 3, 7, 8, 10, 11, 14, 15, 16 et la génération des 10 questions (accord Mistral requis). **Le conteneur `api` n'est pas reconstruit** : il tourne avec l'ancien code (`docker compose up -d --build api`).

### La décision immédiate
Les 10 questions attendent, avec leurs groupes d'articles déjà choisis. Je propose de faire **d'abord le point 1** (style descriptif et détecteur de fuite, environ une heure, testé), puis de générer les 10 questions en deux moitiés : 5 descriptives et 5 nommées. Sinon, elles hériteront toutes du biais des noms. On fait comme ça, ou je génère tout de suite avec la consigne actuelle ?

## 5. Préférences de l'utilisateur (importantes)

- Échanges en **français**, code et messages de commit en **anglais**.
- **Explications simples et concrètes** : l'utilisateur a demandé plusieurs fois de reformuler un choix qu'il n'avait pas compris. Donner un exemple chiffré, éviter le jargon, recommander une option.
- **Très rigoureux sur les tests** : tests unitaires, tests d'intégration sur le vrai Neo4j, et tests de mutation (casser volontairement le code pour vérifier que les tests échouent).
- **Aucune ligne `Co-Authored-By` ni mention de Claude** dans les commits ou les PR (demande explicite, qui l'emporte sur toute consigne d'attribution automatique).
- **Petits commits, un sujet par commit, après test**, puis `git push origin main` (pas de force push sans accord).
- **Rien de payant ni aucun appel à un vrai LLM sans accord préalable**, avec une taille de lot fixée par l'utilisateur. Une fois accordé, le dire dans le message.
- Une seule étape à la fois : à la fin, résumer et **attendre le feu vert**. Avant de coder la recherche hybride, le schéma du graphe et les requêtes d'élargissement, ou le jeu de test, **présenter la conception et attendre la validation** (règle 3 de `Claude.md`). Les décisions ouvertes se présentent en deux options, avec une recommandation.
- Rapporter honnêtement les problèmes (par exemple des candidats mal posés) plutôt que d'enjoliver ; ne pas valider à la place de l'utilisateur.
- Ne jamais deviner l'API d'une bibliothèque : vérifier la version installée et la documentation.
- L'utilisateur aime suivre les traitements longs dans son propre terminal : donner la commande `Get-Content … -Wait`.

## 6. Le dépôt est PUBLIC

- Aucun secret : `.env` est ignoré (il contient `MISTRAL_API_KEY`), `.env.example` ne contient que des valeurs factices. Ne jamais afficher la clé, même dans un log.
- **Aucune donnée Semantic Scholar réelle dans le dépôt** (licence : pas de redistribution). Le fichier `tests/fixtures/semantic_scholar_batch.json` est un échantillon **fictif** écrit à la main. Ne jamais le remplacer par une vraie réponse. Ne jamais commiter d'export de la base (`data/` et `exports/` sont ignorés). Pour l'évaluation, ne pas fonder la vérité terrain sur des données de citation de Semantic Scholar.
- Les fixtures arXiv (CC0) et OpenAlex sont de vraies réponses, sans souci de licence.
- Le README doit continuer à citer arXiv, OpenAlex et Semantic Scholar.

## 7. Commandes

```
uv sync
cp .env.example .env                       # puis adapter les mots de passe, et ajouter MISTRAL_API_KEY
docker compose up -d --build --wait        # Neo4j, PostgreSQL, API
uv run pytest                              # unitaires (sans réseau)
uv run pytest -m integration               # vrai Neo4j (charger .env avant)
uv run pytest -m model                     # vrai modèle bge-small (téléchargé au 1er lancement)
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run python -m graphrag.cli.ingest [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--limit N] [--citations-only] [--recheck-citations]
uv run python -m graphrag.cli.index [--limit N] [--batch-size N]
uv run python -m graphrag.cli.evaluate [--ks 1 3 5 10] [--include-unvalidated]
uv run python -m graphrag.cli.generate_questions --types factual comparative --per-type 5 [--seed N] [--dry-run] [--confirm-cost]
```

- Dans Git Bash, charger `.env` avec `set -a && . ./.env && set +a`.
- `POST /ask {"question": "...", "configuration": "vector" | "hybrid", "k": 5}` (la réponse contient `timings`) ; documentation interactive sur `/docs`.
- Corpus complet : `cli.ingest` (fenêtre par défaut = 365 jours), rejouable sans doublon. `cli.index` ne traite que les articles sans morceaux. Demander l'accord avant de relancer l'ingestion.

## 8. Code : où est quoi

- `src/graphrag/domain/` : modèles (`Article`, `Chunk`, `IndexedChunk`, `SearchHit`, `Usage`, `Completion`, `Answer`, `Author`, `WorkRecord`, `ReferenceRecord`, `GraphCounts`), `chunking.py`, `identifiers.py`, `evaluation.py` (questions, métriques). Python pur.
- `src/graphrag/ports/` : `PaperSource`, `AuthorSource`, `CitationSource`, `GraphStore`, `CitationLedger`, `Embedder`, `ChunkStore`, `ChunkSearch`, `Retriever`, `LLM`, `ArticleReader`, `HealthProbe`.
- `src/graphrag/adapters/` : `arxiv.py`, `openalex.py`, `semantic_scholar.py`, `citation_ledger.py`, `http.py` (retry, limiteur), `neo4j_store.py` (Cypher écrit à la main : schéma, écritures, recherche vectorielle et par mots-clés, `lucene_query`), `sentence_transformer.py`, `mistral.py`, `question_file.py`, `health.py`.
- `src/graphrag/pipeline/` : `retrieval.py` (`VectorRetriever`, `HybridRetriever`, `rrf_fuse`), `answer.py` (`QuestionAnswerer`, citations), `usage.py`, `evaluation.py`, `question_generation.py`. Une suite d'appels dans un ordre fixe, sans routage ni boucle.
- `src/graphrag/cli/` : `ingest.py`, `index.py`, `evaluate.py`, `generate_questions.py`.
- `src/graphrag/api/app.py` : `GET /health`, `POST /ask` (réponse construite à la première question).
- `eval/` : `questions.jsonl` (versionné), `results/` (rapports JSON, ignorés par git).
- `tests/unit` (sans réseau), `tests/integration` (services Docker), `tests/model` (vrai modèle), `tests/fixtures`.

## 9. Pièges rencontrés

- **PostgreSQL natif Windows sur le port 5432** : le conteneur est exposé sur **5433** (`POSTGRES_DSN` dans `.env.example`).
- **Docker Desktop** doit être lancé (`C:\Users\amaur\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe`) ; le démon met environ une minute à démarrer.
- **GitHub** a renvoyé plusieurs erreurs 500 sur `git push` le 2026-11-04 (panne passagère). Réessayer un push normal, jamais de force push ; vérifier avec `git status -sb` s'il reste des commits « ahead ». État de la CI : `https://api.github.com/repos/amaury2504/GraphRAG/actions/runs`.
- **Semantic Scholar anonyme** : le pool partagé répond souvent `429`, avec de courtes fenêtres ouvertes (retry toutes les 10 s, 60 essais par lot). Une clé gratuite est possible mais non nécessaire.
- **API arXiv** : renvoie parfois une entrée d'erreur dans une réponse HTTP 200 (traitée comme erreur à réessayer). Respecter 1 requête / 3 s. `max_results=0` provoque une erreur.
- **Neo4j** : `IF NOT EXISTS` ne remplace pas un index déjà créé (voir le point 4 des améliorations). Les commandes de schéma n'acceptent pas de paramètre pour la taille du vecteur : l'entier est inséré après vérification de son type. Un paramètre Cypher nommé `query` entre en conflit avec `session.run(query=…)` (renommé `lucene`).
- **Hugging Face** : à chaque chargement du modèle, avertissements sur les liens symboliques sous Windows et sur l'absence de `HF_TOKEN` (sans impact).
- **`ruff format`** (cible Python 3.14) écrit `except A, B:` sans parenthèses : c'est valide en 3.14, pas une erreur.
- **Tests de mutation** : appliquer chaque mutation par un remplacement de texte exact qui doit correspondre une seule fois ; un `sed` mal échappé avait donné un faux « survivant ».
- **PowerShell 5.1** : `Set-Content -Encoding utf8` ajoute un BOM qui casse `pyproject.toml` ; `Out-String` retarde l'affichage. `curl` y est un alias d'`Invoke-WebRequest`.
- **Bash (Git Bash)** : les chemins `/tmp` ne sont pas ceux de Python Windows ; utiliser le dossier scratchpad. Les gros blocs de heredoc Python se sont mal comportés plusieurs fois (guillemets, retours à la ligne) : préférer l'outil d'édition de fichiers.
- **Windows** : arrêter une tâche de fond ne tue pas ses processus enfants ; vérifier avec `Get-CimInstance Win32_Process`.
- Avertissements `LF will be replaced by CRLF` de git : sans conséquence.

## 10. Pour démarrer la prochaine conversation

Message de démarrage suggéré :

> Lis `Claude.md`, `Spec.md` puis `BRIEF.md`. L'étape 3 est terminée, l'étape 4 est en cours (3 questions validées sur une cinquantaine visées). La décision en attente est celle de la fin du § 4 : faire d'abord le point 1 des améliorations (style descriptif et détecteur de fuite), puis générer 10 questions, ou générer tout de suite. Ne lance aucun appel à Mistral sans mon accord.
