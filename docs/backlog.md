# Backlog

États : libre, en cours, bloqué, fait.

## À faire

| Tâche | Contenu | Dépend de | État |
| --- | --- | --- | --- |
| T2 | Versioning atomique : chunks immuables partagés entre versions, manifeste par version, pointeur « version courante » basculé en une opération, rollback | T1 | libre |
| T3 | Cohérence en lecture : statuts de version (`pending`, `indexed`, `active`), la version n'est lisible qu'une fois vecteurs et texte intégral indexés | T2 | libre |
| T4 | Garbage collection (mark and sweep) des chunks non référencés par une version conservée, durée de rétention, compactage ou reconstruction de l'index vectoriel | T2 | libre |
| T5 | Versioning du modèle d'embedding : `embedding_model` et version sur chaque vecteur, nouvel index construit en parallèle, bascule atomique | T2, T3 | libre |
| T7 | Balayage de la taille de chunk et de l'overlap sur `eval/` (recall@k, MRR) ; relancer l'évaluation après chaque changement de chunker ou de modèle | T1 | libre |

## Fait

| Tâche | Ce qui a été livré |
| --- | --- |
| T1 | (PR #4) Ré-ingestion incrémentale : hash d'article (titre + résumé + `CHUNKER_VERSION`), l'ingestion réécrit seulement les articles nouveaux ou modifiés et les compte, l'indexation ré-indexe les articles dont le hash diffère (chunks, orphelins supprimés et `indexed_hash` dans une transaction), rejeu idempotent, reprise après interruption. Identifiants de chunk inchangés (`arxiv_id#position`). Le modèle d'embedding n'est pas dans le hash (T5). |
| T6 | Graphe dérivé à jour : `upsert_works` et `upsert_citations` remplacent les relations de l'article (les `AUTHORED_BY` et `CITES` retirés à la source disparaissent, les auteurs sans article aussi, dans la même transaction) ; `GraphStore.delete_article` (chunks, `CITES` entrants et sortants, auteurs orphelins, sans CLI) ; `ingest` oublie dans le registre de citations les articles modifiés pour ré-interroger leurs références. Hors tâche : extraction LLM `Méthode`/`Benchmark` (aucune entité ni résumé dérivé des chunks n'existe encore). |
