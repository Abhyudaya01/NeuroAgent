# Fresh ten-question evaluation

On November 17, 2026, the revised agent was evaluated on FRAMES questions `frames_0056` through `frames_0065`. These questions lie outside the original 50-question pilot and the five reserved follow-up questions. They were selected by dataset position from the same cached FRAMES release before execution. The cached first 55 questions were checked against the earlier inputs.

Both configurations shared the current executor, evidence resolver, supported-answer gates, numerical execution path, and subtask-scoped prompts. The ReAct-style baseline disabled the critic, critic retries, and local retrieval memory. Full NeuroAgent enabled those components. The code was fixed for both agent runs. Each configuration attempted all ten questions using GPT-4o, with a $1 estimated agent chat-cost stop and credit reserved for grading.

| Configuration | Initial automated grade | Reviewed answers correct | Mean seconds/question | Agent chat cost | Mean retries/question |
| --- | ---: | ---: | ---: | ---: | ---: |
| ReAct-style baseline | 5/10 | 6/10 | 14.3 | $0.182 | 0.0 |
| Full NeuroAgent | 4/10 | 6/10 | 19.9 | $0.407 | 2.5 |

Both configurations answered the same six questions correctly after review. Full NeuroAgent used 2.24 times the estimated agent chat cost and 1.39 times the observed per-question elapsed time. The difference combines the critic, retry policy, and retrieval memory. Full NeuroAgent planned one document-retrieval task against the technical seed corpus.

All 20 attempts completed through the runner without operational exceptions. Seven baseline plans and five full-agent plans completed every planned task. A partially completed plan may still contain the requested answer. The four incorrect questions in each configuration concerned cancer-survivorship evidence, a multi-part dinosaur reconstruction answer, a championship count, and a film-runtime comparison. The full-agent dinosaur response identified the requested species but missed part of the required anatomical description; the baseline selected the wrong species.

The initial GPT-4o grades were 5/10 for the baseline and 4/10 for full NeuroAgent. Three decisions were reviewed against the saved report text under the existing equivalence and assertion rules. Both reports explicitly supplied the correct historical-event order as numbered lists. Their comma-joined predicted answers failed the original witness formatting check. The reviewed grades validate each reported item in the asserted order. Raw grades and reviewed grades are stored separately.

Full NeuroAgent answered the food-colorant question with beet red. The initial judge rejected this because the reference was betanin. The E162 specification names beetroot red, betanin and lists Beet Red as a synonym in the food-additive context. The answer was accepted under the rubric’s equivalent-name rule. The source check supports this food-colorant naming equivalence. [Official E162 specification](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02012R0231-20261108)

Grading implementation was improved after agent execution to represent ordered list items explicitly in `answer_parts`. The witness check requires each item to occur in order in the quoted report span, and still requires a reported answer anchor. It rejects a computed answer absent from the report even if its numerical inputs are present. All 80 tests pass, including list-order preservation and prevention of missing-result inference. These grading changes and the source-supported synonym review do not alter the saved agent responses.

This is a small, single-pass pilot with post-run model-assisted grading review. It establishes no general accuracy advantage for the combined critic-enabled configuration. The original 50-question re-audit and this ten-question pilot involve different question sets and agent versions, so they remain separate results. Both runs were concurrent, and elapsed times include shared provider and tool latency. A complete human audit is outstanding.

Additional estimated chat spending was **$0.69**, comprising $0.589 for the agents and $0.104 for grading. Cumulative estimated chat spending is **$11.91**, implying approximately **$1.70** remains from the reported $13.61 starting balance. Cached input is priced at the full input rate. Embeddings, Tavily, E2B, and unrelated account activity are excluded. [GPT-4o pricing](https://developers.openai.com/api/docs/models/gpt-4o)

The shared execution source fingerprint was `ed2f1f1eb6697e2397f0876746930536f571546ce9c08a28041b4bb4c282f337`. Input and memory fingerprints, package versions, raw per-question results, original grades, review records, and source-run hashes are retained with the artifacts. The review records identify each changed verdict and its supporting evidence.

[Comparison and cost accounting](../eval/results/frames_fresh_10/comparison.json) · [Reviewed baseline grades](../eval/results/frames_fresh_10/react_baseline_grades_reviewed.json) · [Reviewed full-agent grades](../eval/results/frames_fresh_10/neuro_full_grades_reviewed.json) · [Earlier 50-question re-audit](FRAMES_REAUDIT.md)

The per-question files are available locally under the Git-ignored `eval/results/frames_fresh_10/` directory. This document preserves the aggregate results for the repository.
