# Backlog

États : libre, en cours, bloqué, fait.

## À faire

| Tâche | Contenu | Dépend de | État |
| --- | --- | --- | --- |
| T3 | Cohérence en lecture : statuts de version (`pending`, `indexed`, `active`), la version n'est lisible qu'une fois vecteurs et texte intégral indexés | T2 | libre |
| T4 | Garbage collection (mark and sweep) des chunks non référencés par une version conservée, durée de rétention, compactage ou reconstruction de l'index vectoriel. Reprend la suppression des chunks orphelins que T1 faisait et que T2 a retirée, y compris les anciens chunks à clé `arxiv_id#position` | T2 | libre |
| T5 | Versioning du modèle d'embedding : `embedding_model` et version sur chaque vecteur, nouvel index construit en parallèle, bascule atomique | T2, T3 | libre |
| T7 | Balayage de la taille de chunk et de l'overlap sur `eval/` (recall@k, MRR) ; relancer l'évaluation après chaque changement de chunker ou de modèle | T1 | libre |

## Fait

| Tâche | Ce qui a été livré |
| --- | --- |
| T1 | (PR #4) Ré-ingestion incrémentale : hash d'article (titre + résumé + `CHUNKER_VERSION`), l'ingestion réécrit seulement les articles nouveaux ou modifiés et les compte, l'indexation ré-indexe les articles dont le hash diffère (chunks, orphelins supprimés et `indexed_hash` dans une transaction), rejeu idempotent, reprise après interruption. Identifiants de chunk inchangés (`arxiv_id#position`). Le modèle d'embedding n'est pas dans le hash (T5). |
| T6 | Graphe dérivé à jour : `upsert_works` et `upsert_citations` remplacent les relations de l'article (les `AUTHORED_BY` et `CITES` retirés à la source disparaissent, les auteurs sans article aussi, dans la même transaction) ; `GraphStore.delete_article` (`CITES` entrants et sortants, auteurs orphelins, sans CLI ; les chunks, immuables, sont seulement détachés et laissés à T4) ; `ingest` oublie dans le registre de citations les articles modifiés pour ré-interroger leurs références. Hors tâche : extraction LLM `Méthode`/`Benchmark` (aucune entité ni résumé dérivé des chunks n'existe encore). |
| T2 | (PR #6) Versioning atomique : clé de chunk par contenu (`arxiv_id#position#empreinte`), chunks immuables jamais supprimés (la suppression des orphelins passe à T4), version = instantané du corpus (nœuds `Version`, relations `CONTAINS`, `PREVIOUS`, pointeur `Current`) publié en fin d'`index`, manifeste et pointeur changés dans une transaction, recherches limitées à la version courante (pas de filtre sans version publiée), `index --rollback` répétable. Le prochain `index` sur une base sans hash réindexe tout puis publie la version 1. |
