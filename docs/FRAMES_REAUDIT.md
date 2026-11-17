# FRAMES grading re-audit

On November 17, 2026, all 150 saved responses from the original three-configuration FRAMES pilot were regraded with a quotation-backed GPT-4o rubric. The saved agent runs and their latency, retries, and tool usage were retained. This audit measures those earlier agent versions.

| Configuration | Original unquoted grade | Re-audited positive answers | Re-audited score |
| --- | ---: | ---: | ---: |
| ReAct-style baseline | 15/50 | 9/50 | 18% |
| NeuroAgent without memory | 7/50 | 6/50 clear, 1 borderline | 12–14% |
| Full NeuroAgent | 7/50 | 7/50 | 14% |

The strict automated score for the no-memory variant is 12%. Its `frames_0032` report includes the relation Dwight (David) Eisenhower after researching the shared death date, but the grader found no clearly selected final answer. The explanation acknowledges the correct relation while describing a focus on an irrelevant president. This is a borderline application of the explicit-answer rule. Counting it correct raises that configuration to 14%. It remains flagged for adjudication.

Six baseline positives were reclassified as incorrect. They include an unperformed rounding task (`frames_0006`), a missing Olympic-team count (`frames_0008`), an unperformed year difference (`frames_0019`), an uncomputed arithmetic result (`frames_0029`), a list of prime-minister terms without the requested hypothetical conclusion (`frames_0030`), and an unselected car name in a list (`frames_0042`). These decisions distinguish assertions from data that could be used to calculate or infer an answer.

The baseline Michael Richards answer (`frames_0036`) was retained after correcting a quotation-format failure. The original report explicitly identifies him as the oldest actor. Both NeuroAgent Tommy Lawton answers (`frames_0028`) were also retained after recovering an actual source span with a typographic apostrophe. Full NeuroAgent’s two-station answer (`frames_0045`) was retained because both requested names occur in the quoted report passage. No reference answer was inserted into a report.

The grader first evaluates answer equivalence against the gold answer. Positive grades must also have a supporting report span containing each asserted answer component. The source-span matcher preserves words, signed numbers, and decimal values while tolerating punctuation formatting, and stores the actual recovered report text. It therefore rejects a predicted 506000 supported only by a reported 506,132. One format-correction retry is allowed. In this audit, five initially rejected quotation cases received focused model rechecks. The remaining formatting failures were resolved mechanically against the original reports. All 22 clear positive answers have source-span witnesses verified against their saved reports.

Original grading artifacts, the first quotation-backed pass, format rechecks, and final source-span-reviewed grades are retained separately. Each grade file records the source-run hash and rubric identity. The reviewed artifacts add parent-grading hashes, validation-policy versions, repair records, and token usage. The aggregate artifact verifies the same 50 ordered question IDs across configurations and source hashes against the original run bytes.

Full NeuroAgent is four percentage points below the baseline on this strict single-pass evaluation. The critic-enabled no-memory variant is four to six points below the baseline, depending on the borderline decision. Full NeuroAgent is zero to two points above the no-memory variant. These are descriptive differences on 50 questions. The source runs used independently generated plans and concurrent execution. They do not isolate a causal memory or critic benefit, and the current evidence-gated agent still needs a larger fresh evaluation.

This audit used an estimated **$1.92** in chat costs, including format rechecks. Cumulative estimated chat spending is **$11.22**, implying approximately **$2.39** remains from the reported $13.61 starting balance. Cached input is priced at the full input rate. The ledger excludes embeddings, external-tool charges, and unrelated account activity. [GPT-4o pricing](https://developers.openai.com/api/docs/models/gpt-4o)

All 78 tests pass. Added regressions cover a malformed quotation recovering through one retry, typographic punctuation, multi-part witnesses, missing numerical results, and protection against grading a rounded result that the report never asserted. These remain automated grades with targeted model-assisted inspection. A complete human audit is outstanding.

[Audited comparison, changed verdicts, and cost accounting](../eval/results/frames_budget_50/comparison_audited.json) · [ReAct-style grades](../eval/results/frames_budget_50/react_baseline_grades_audited_v3.json) · [No-memory grades](../eval/results/frames_budget_50/neuro_no_memory_grades_audited_v3.json) · [Full-agent grades](../eval/results/frames_budget_50/neuro_full_grades_audited_v3.json)

Per-question artifacts are available locally in the Git-ignored results directory. This document preserves aggregate results and limitations for the repository.
