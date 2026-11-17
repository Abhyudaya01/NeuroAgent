# FRAMES pilot results

The 150 saved responses were re-audited on November 17, 2026. The table below uses
quotation-backed grades. The no-memory score has one borderline case, yielding a
12–14% range. See [FRAMES_REAUDIT.md](FRAMES_REAUDIT.md) for changed verdicts and limitations.

On November 17, 2026, all three configurations were run on the same first 50 prepared FRAMES questions. Each used GPT-4o for planning, synthesis, query reformulation, and critique where enabled. A separate GPT-4o judge graded the saved reports against the gold answers. Each configuration and its judge completed all 50 attempts.

| Configuration | Correct | Judge accuracy | Mean seconds/question | Mean chat tokens/question | Mean retries/question | Agent chat cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ReAct-style baseline | 9/50 | 18% | 12.2 | 1,502 | 0.00 | $0.34 |
| NeuroAgent without memory | 6/50 clear + 1 borderline | 12–14% | 30.2 | 17,522 | 5.54 | $3.01 |
| Full NeuroAgent | 7/50 | 14% | 30.7 | 16,596 | 5.38 | $2.78 |

Full NeuroAgent scored four percentage points below the baseline, with 2.51 times its observed latency and 8.13 times its estimated agent chat cost. Adding the critic and its retry policy without memory reduced the quoted-answer score by four to six points. Enabling retrieval memory changed the score by zero to two points. These are descriptive single-pass comparisons with independently generated plans.

The full run planned only four document-retrieval tasks across 50 questions. Its persistent FAISS corpus contains technical agent notes, while these FRAMES questions require general factual and historical knowledge. This run offers limited evidence about the benefit of retrieval memory.

Every successfully returned report had nonempty outputs for all planned tasks. The no-memory run had one operational exception on question `frames_0027`, where the model omitted a required critic field. That attempt counts as incorrect, giving an operational error rate of 2%. The other configurations had zero operational errors. The 49 returned no-memory reports and all 50 reports in each other configuration reached the end of their plans.

Full NeuroAgent had 20 of 43 incorrect answers with a mean task-level critic score of at least 0.7. Under strict grading, the no-memory variant had 21 of 44 incorrect answers above that threshold. The borderline case changes its figures to 20 of 43. These quality scores across subtasks flag high-scoring wrong reports and do not establish calibrated probabilities.

Spot checks found unresolved entity identification, numeric questions left unanswered, and generic analysis templates in place of calculations. Some fallback templates discuss retrieval, hierarchical memory, and reliability even for unrelated factual questions. Nonempty task outputs therefore provide a weak completion metric. Answer presence alone also admits incidental matches, especially for numeric gold answers.

The main experiment includes the tool-history reset, preservation of failed task status, saved critic scores, and web-query reformulation on retries. A separate one-question smoke comparison used 12 retries and $0.119 before web reformulation, then four retries and $0.055 after it. The gold answer was absent in both. Different model and search outputs prevent attributing that cost difference solely to the code change.

After the main experiment, the redundant critic `should_retry` field received a default because the application recomputes retry decisions locally. A regression test checks that an omitted field still allows low-scoring tasks to retry. A separate rerun of `frames_0027` completed without an operational error for $0.081. It remains separate from the headline benchmark. Answer correctness of that follow-up has not been graded. At completion of this experiment, the test suite passed 51 tests.

At the end of the original experiment, estimated main agent chat spending was $6.129, with $1.698 for grading. Including smoke tests and the parsing-fix verification, estimated chat spending was **$8.11**, implying approximately **$5.50** of the reported $13.61 starting balance at that stage. Later follow-up and audit spending are recorded in [FRAMES_REAUDIT.md](FRAMES_REAUDIT.md). This is an estimate from recorded chat tokens. Cached input is priced at the full input rate, and embeddings, Tavily, E2B, and unrelated account activity are outside this accounting. Published GPT-4o rates were $2.50 per million input tokens and $10.00 per million output tokens when checked. [Official model pricing](https://developers.openai.com/api/docs/models/gpt-4o)

All three configurations ran concurrently. Per-question elapsed times include provider and tool latency under that shared workload. Web evidence and model responses can vary between runs. This pilot covers 50 of the 824 FRAMES questions, and its automated grades received spot checks. The headline figures have not received a complete human audit. The original 30% baseline score was revised to 18% by quotation-backed grading.

The main runs share source fingerprint `812bc897972275afcd89268d35ebec96b634597e97540d64cfe5e404e1622def`. They used a modified working tree based on Git revision `34bc407356a23427685cb27e904937df122ae0c9`. The critic-field default was added afterward. Source, input and memory fingerprints, package versions, task outputs, model token counts, and grading explanations are saved with the artifacts.

[Audited comparison and provenance](../eval/results/frames_budget_50/comparison_audited.json) · [Failure analysis](../eval/results/frames_budget_50/failure_analysis.json) · [Evaluation instructions](FRAMES_EVALUATION.md)

The per-question run and grading files live in `eval/results/frames_budget_50/`. That directory is ignored by Git and is available in this checkout. This document retains the aggregate results for the repository.
