# Backlog

États : libre, en cours, bloqué, fait.

## À faire

| Tâche | Contenu | Dépend de | État |
| --- | --- | --- | --- |
| T1 | Ré-ingestion incrémentale : identifiants de chunk déterministes (document + position), content hash (texte + version du chunker + modèle d'embedding), ajout/modification/suppression des seuls chunks changés, idempotence et reprise sur événement rejoué | | libre |
| T2 | Versioning atomique : chunks immuables partagés entre versions, manifeste par version, pointeur « version courante » basculé en une opération, rollback | T1 | libre |
| T3 | Cohérence en lecture : statuts de version (`pending`, `indexed`, `active`), la version n'est lisible qu'une fois vecteurs et texte intégral indexés | T2 | libre |
| T4 | Garbage collection (mark and sweep) des chunks non référencés par une version conservée, durée de rétention, compactage ou reconstruction de l'index vectoriel | T2 | libre |
| T5 | Versioning du modèle d'embedding : `embedding_model` et version sur chaque vecteur, nouvel index construit en parallèle, bascule atomique | T2, T3 | libre |
| T6 | Mise en cohérence du graphe dérivé (relations, entités ou résumés issus des chunks modifiés ou supprimés) | T1 | libre |
| T7 | Balayage de la taille de chunk et de l'overlap sur `eval/` (recall@k, MRR) ; relancer l'évaluation après chaque changement de chunker ou de modèle | T1 | libre |

## Fait

| Tâche | Ce qui a été livré |
| --- | --- |
