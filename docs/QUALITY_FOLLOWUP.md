# Answer-quality follow-up

The original 50-question pilot has now been re-audited. These five-question
follow-up scores still use the earlier grading rubric and remain provisional.
See [FRAMES_REAUDIT.md](FRAMES_REAUDIT.md) for the completed pilot re-audit.

On November 17, 2026, the numerical execution and dependency handling paths were revised after the 50-question FRAMES pilot. The same GPT-4o agent settings and grading rubric were used for this follow-up.

Plain-English computation tasks now generate Python using the supplied evidence and execute it in E2B. The previous scripts printed generic research templates without performing the requested calculation. Missing inputs now produce an explicit failed result. Sandbox failure confidence is preserved. Successful code execution by itself does not establish the factual correctness of its inputs.

Dependent searches receive earlier task findings and the original question when resolving their queries. Explicit dependencies are restricted to earlier tasks. Synthesis receives the original constraints and retry feedback, and asks for the requested answer or calculation. Critique receives the original question and dependency evidence and checks entity, ordinal, time, unit, and computation requirements.

Five previously failed questions were selected for development because they exercise entity resolution and numerical reasoning. They were chosen from the original saved failures before this rerun. Gold answers were used only by the grader. The next five dataset questions were reserved for a separate check, and the code was frozen for both checks.

| Evaluation | Correct | Agent chat cost | Mean retries/question |
| --- | ---: | ---: | ---: |
| Selected failures, prior full agent | 0/5 | $0.210 | 4.4 |
| Selected failures, revised full agent | 3/5 | $0.255 | 1.8 |
| Reserved questions, revised baseline | 4/5 | $0.060 | 0.0 |
| Reserved questions, revised full agent | 3/5 | $0.289 | 1.8 |

The revised agent recovered `frames_0001` (Jane Ballou), `frames_0003` (87 years), and `frames_0006` (506000). It still identified the wrong city in `frames_0019`, producing mathematically consistent arithmetic from the wrong factual input. In `frames_0027`, it reported the full-moon dates needed for the calculation as missing. It no longer emitted a memory-research template for that calculation.

The reserved questions were `frames_0051` through `frames_0055`, outside the original 50-question benchmark. The revised full agent answered questions 52, 53, and 54 correctly. It missed the product discontinued by the Alaska Commercial Company and the film-title letter-count question. The baseline answered four of the five correctly. This check continues to show no accuracy advantage for the critic loop.

Both reserved runs and the development run completed all five questions without operational exceptions. All 56 tests pass, including regressions for executed arithmetic, missing calculation inputs, sandbox failures, dependency query resolution, exclusion of future-task findings, and original constraints reaching synthesis and critique.

Additional estimated chat spending was **$0.76**, including all three runs and their graders. Cumulative estimated chat spending is **$8.86**, implying approximately **$4.75** remains from the reported $13.61 starting balance. Estimates price cached input at the full input rate and exclude embeddings, external tool charges, and unrelated account activity. [GPT-4o pricing](https://developers.openai.com/api/docs/models/gpt-4o)

The development set was selected from failures, and each check contains only five questions. Several prompts and execution paths changed together. Model and search variability also affect comparisons. The old development run shared resources with other configurations; the revised development run ran alone. Timing differences therefore do not isolate an optimization. The original 50-question accuracy figures remain the recorded pilot results. A fresh larger evaluation of the revised code remains outstanding.

The follow-up source fingerprint is `98f8b576d1721bf9b783784f414ed1bcd704c16fa198cab18f944eb0eecb1802`. Each saved run includes input, source, and memory fingerprints plus model and package versions. The comparison checks that grading source hashes match the saved run files.

[Follow-up metrics and provenance](../eval/results/frames_quality_holdout/comparison.json) · [Original 50-question results](FRAMES_RESULTS.md)

Per-question outputs and grading explanations are stored locally under `eval/results/frames_quality_dev/` and `eval/results/frames_quality_holdout/`. Those directories are ignored by Git. This document preserves the aggregate results for the repository.
