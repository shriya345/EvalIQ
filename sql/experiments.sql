-- Which retrievers cover reference passages, and how long do searches take?
SELECT dataset_digest,method,COUNT(recall) AS labeled_cases,AVG(recall) AS mean_recall,AVG(retrieval_ms) AS mean_retrieval_ms FROM rag_cases GROUP BY dataset_digest,method;

-- Which question categories produce incorrect generated answers?
SELECT experiment_id,category,COUNT(correct) AS assessed,AVG(1.0-correct) AS failure_rate FROM rag_cases GROUP BY experiment_id,category;

-- How do quality and observed token usage compare across runs?
SELECT experiment_id,COUNT(*) AS cases,AVG(1.0*correct) AS correctness,COUNT(tokens) AS known_token_counts,AVG(tokens) AS mean_tokens FROM rag_cases GROUP BY experiment_id;

-- How does recall change across successive runs on the same version of the corpus?
WITH totals AS (SELECT dataset_digest,experiment_id,created,AVG(recall) AS recall FROM rag_cases GROUP BY dataset_digest,experiment_id,created) SELECT *,recall-LAG(recall) OVER (PARTITION BY dataset_digest ORDER BY created,experiment_id) AS recall_delta FROM totals ORDER BY created;
