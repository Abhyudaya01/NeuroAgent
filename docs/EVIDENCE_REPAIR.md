# Evidence gate and prerequisite repair

On November 17, 2026, the executor gained an explicit research-answer contract. Research output is converted into a direct subtask answer with source passages, or an explicit evidence gap. Citation source identifiers must match the retrieved sources. Quoted words and signed numeric values are matched against the cited content, tolerating punctuation formatting and retaining the actual source span. The source text is retained in task state for audit. Semantic entailment still depends on the resolver and critic.

An unresolved prerequisite blocks dependent work. The critic can revisit that prerequisite once, increments its retry count, and invalidates transitive dependent results. Unrelated completed tasks are preserved and skipped. A calculation that needs an additional fact can make a targeted search on its first own retry. Each research step can fetch relevant passages from its top two sources on the first retry. Retrieval source passages retain their URL or document attribution.

An evidence gap forces a zero critic score. An answer rejected at the end of its retry budget remains unresolved. The global transition budget returns a partial report instead of allowing unbounded repair. Reports and benchmark `completed_tasks` now require completed status and nonempty output. The separate `output_tasks` field retains the earlier nonempty-output count. Failure analysis distinguishes unfinished tasks from missing text.

Local graph integration tests exercise prerequisite repair, transitive invalidation, independent-task preservation, and bounded termination. Additional tests cover wrong-source quotations, fabricated quotes, numeric mismatches, missing facts, source-extraction failures, short factual passages, and subtask boundaries. All 74 tests pass.

The live development checks used three saved failures, `frames_0019`, `frames_0027`, and `frames_0055`. They showed bounded prerequisite rechecks in saved task state. The city-ranking and blue-moon questions still lacked required historical evidence and ended with blocked downstream tasks. They remain unresolved.

Two implementation issues were found during these checks. A strict quotation comparison rejected a valid film-debut passage because of Wikipedia-link punctuation. Matching now recovers the original source span with identical words and numeric values. An intermediate film-title lookup also demanded the final letter count. The resolver, query writer, and critic now explicitly evaluate only the current subtask while using the original question for constraints.

After the boundary correction, `frames_0055` completed four tasks with David Arnold, The Young Americans, and a computed result of 17 letters. The corrected grader verified the exact report assertion. This is a one-question development verification, with no new full-benchmark or unseen accuracy claim.

A grading false positive was also reproduced. The original judge marked an earlier blocked film report correct by claiming it stated 17. The report did not contain that assertion. The grader now requires an exact `answer_quote` from the report, with the predicted answer appearing in that quotation. Positive grades lacking this witness are rejected mechanically. Its rubric fingerprint changes, so prior grading checkpoints cannot be silently resumed under the new policy.

Regrading the three pre-boundary-correction responses with the quotation-backed grader produced 0/3 correct. The subsequent single film-question verification produced 1/1. The original unquoted grade is preserved alongside the corrected artifact. The original 50-question scores have subsequently been re-audited in [FRAMES_REAUDIT.md](FRAMES_REAUDIT.md). The earlier five-question follow-up scores still use the older rubric and remain provisional.

Additional estimated chat spending for this phase was **$0.43**, including intermediate checks, a diagnostic replay, and grading. Cumulative estimated chat spending is **$9.30**, implying approximately **$4.31** remains of the reported $13.61 starting balance. Cached input is priced at the full input rate. Embeddings, Tavily search/extraction, E2B, and unrelated account activity are excluded. [GPT-4o pricing](https://developers.openai.com/api/docs/models/gpt-4o)

The final agent verification used source fingerprint `2ff14ca307ebf58503d0826db0e03246b712d4f9ed9192374bd58008769751fd`. The quotation-backed grader was added afterward and carries its own rubric fingerprint. [Cost accounting](../eval/results/frames_evidence_boundary/summary.json) · [Final run](../eval/results/frames_evidence_boundary/neuro_full_latest.json) · [Quotation-backed grade](../eval/results/frames_evidence_boundary/neuro_full_grades_quoted.json)

Per-question artifacts remain local under the Git-ignored `eval/results/frames_evidence_*` directories.
