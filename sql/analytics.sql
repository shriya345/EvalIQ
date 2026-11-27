-- How many requests started per UTC day?
SELECT substr(timestamp,1,10) AS day, count(*) AS requests FROM telemetry_events WHERE event_type='request' GROUP BY 1 ORDER BY 1;

-- Which applications encounter errors during a task?
WITH tasks AS (SELECT application,trace_id,max(CASE WHEN status='error' OR event_type='error' THEN 1 ELSE 0 END) AS failed, max(CASE WHEN event_type='request' THEN 1 ELSE 0 END) AS started FROM telemetry_events GROUP BY 1,2) SELECT application,avg(1.0*failed) AS error_rate FROM tasks WHERE started=1 GROUP BY 1;

-- Which models have slow average response times, and how many observations support the mean?
SELECT model,avg(latency_ms) AS mean_ms,count(latency_ms) AS observed FROM telemetry_events WHERE event_type='response' GROUP BY 1;

-- What is the tail response latency per model?
SELECT model,percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms FROM telemetry_events WHERE event_type='response' GROUP BY 1;

-- What is known about spend per successful task when every started task has a priced outcome?
WITH terminal AS (SELECT *,row_number() OVER (PARTITION BY application,trace_id ORDER BY timestamp DESC,event_id DESC) AS rn FROM telemetry_events WHERE event_type='response'), tasks AS (SELECT DISTINCT application,trace_id FROM telemetry_events WHERE event_type='request') SELECT t.application,count(r.estimated_cost_usd) AS priced,count(*) AS started,CASE WHEN count(*)=count(r.estimated_cost_usd) THEN sum(r.estimated_cost_usd)/nullif(sum(CASE WHEN r.status='ok' THEN 1 ELSE 0 END),0) END AS usd_per_success FROM tasks t LEFT JOIN terminal r ON t.application=r.application AND t.trace_id=r.trace_id AND r.rn=1 GROUP BY 1;

-- How does recorded response latency vary across prompt versions?
SELECT application,model,prompt_version,count(*) AS responses,avg(latency_ms) AS mean_ms FROM telemetry_events WHERE event_type='response' GROUP BY 1,2,3;

-- Which evaluation runs pass their cases?
SELECT r.id,r.dataset,r.variant,count(*) AS cases,avg(1.0*v.passed) AS pass_rate FROM evaluation_runs r JOIN evaluation_results v ON r.id=v.run_id GROUP BY 1,2,3;

-- Which tools need investigation?
SELECT tool_name,count(*) AS calls,sum(CASE WHEN status='error' THEN 1 ELSE 0 END) AS failures FROM telemetry_events WHERE event_type='tool_call' GROUP BY 1 ORDER BY failures DESC;

-- How does volume trend over the last three observed days?
WITH daily AS (SELECT substr(timestamp,1,10) AS day,count(*) AS n FROM telemetry_events WHERE event_type='request' GROUP BY 1) SELECT day,n,avg(n) OVER (ORDER BY day ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rolling_avg FROM daily ORDER BY day;

-- Which recorded failure categories are most frequent within each application?
WITH counts AS (SELECT application,coalesce(error_type,event_type) AS category,count(*) AS failures FROM telemetry_events WHERE status IN ('invalid','error') OR event_type='error' GROUP BY 1,2) SELECT *,dense_rank() OVER (PARTITION BY application ORDER BY failures DESC) AS frequency_rank FROM counts ORDER BY application,frequency_rank;
