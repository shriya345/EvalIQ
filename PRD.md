# Product requirements

**Users.** An engineer investigates failed agent steps. An analyst compares answer quality with latency and cost. A product owner decides whether a prompt change warrants further testing.

**Problem.** Request logs show execution, while review spreadsheets contain reference answers. Teams need to inspect both and understand which metrics have enough observations to support a decision.

| Workflow | Acceptance signal | Priority |
| --- | --- | --- |
| Import a JSONL export after each run | Existing IDs remain unique and malformed lines are reported | P0 |
| Score candidate answers against fixed cases | Each failure identifies the failed check and case | P0 |
| Compare two configurations | Same-case set and missing metric coverage are visible | P0 |
| Inspect a slow or failed trace | Root, agent steps, timestamps and parent IDs are available | P0 |
| Integrate an application programmatically | A Python client and local API use the same contract | P1 |

Track correctness and source coverage per case, successful tasks per started trace, P95 duration and known expenditure. The test suite checks behavioral acceptance. There is no user-adoption or ROI evidence. Single-user, local operation and small benchmark datasets define this release. Public deployment, access control, distributed tracing and subjective semantic judging are outside scope.
