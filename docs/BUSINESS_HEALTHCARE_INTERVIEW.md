# Model choice, business value, and healthcare interview preparation

These sample answers connect NeuroAgent to practical agentic AI engineering. Adapt personal motivations and ownership statements to your actual experience. The healthcare workflow below is a proposed extension, not an implemented or clinically validated capability of NeuroAgent.

## Why did you use these models?

> The saved NeuroAgent comparisons use GPT-4o. Keeping the model consistent across workflow configurations helped reduce experimental variation while studying the critic loop and retrieval memory. I have not established that GPT-4o is the best model across all providers.
>
> For a company deployment, I would compare candidate models on representative company tasks. I would measure answer quality, critical errors, tool-call reliability, latency, approved data handling, and total cost per successfully completed task. I would choose the least expensive option that meets the agreed requirements.

The planner and critic are configurable through `OPENAI_CHAT_MODEL`; their code fallback is GPT-3.5 Turbo. Other reasoning paths use GPT-4o, and embeddings use `text-embedding-3-small`. Verify run metadata when discussing a specific experiment.

For cost control, test a smaller model for extraction, classification, or routing. Use ordinary code for fixed rules and calculations. Escalate difficult cases based on observable evidence gaps, failed validation, or risk classification. Validate model confidence against correctness before using it for routing. An open-weight deployment comparison should include GPU, maintenance, security, and operational costs.

## How would you improve a business while controlling cost?

> I would identify a frequent, expensive workflow and measure its current processing time, error rate, and review effort. Then I would build a narrow pilot and compare the total cost and quality against the existing process.

Use this metric:

**Cost per accepted task = total operating and review cost / tasks completed to the required standard.**

Include model calls, embeddings, retrieval, external tools, infrastructure, monitoring, human review, and rework. Evaluate financial benefit with the business owner, including integration and ongoing support costs.

For an illustrative referral workflow with 1,000 cases per month, reducing preparation time from six minutes to three minutes, including review, releases 50 staff-hours per month. This is a hypothetical calculation, not a measured NeuroAgent outcome. Establish how that capacity creates business value before treating it as a financial saving.

NeuroAgent provides an example of why this measurement matters. Its original full configuration took about 4.1 times the baseline runtime and averaged 7.42 task retries per question. This measures the complete configuration's end-to-end runtime difference; it does not isolate the critic's causal overhead. The experiments have not established a consistent accuracy benefit from self-correction.

Candidate improvements include conditional critique, bounded retries, stopping when evidence remains unavailable, compact context, privacy-aware document caching, and concurrent execution of independent operations. Measure the resulting quality and cost before adopting each change.

## How could an agent help in healthcare?

A proposed starting workflow is preparing a referral packet for staff review.

1. Retrieve authorized records for the correct patient and encounter.
2. Check which documents the referral requires.
3. Extract relevant facts with supporting sources and dates.
4. Flag missing or conflicting information.
5. Draft the packet for an appropriate reviewer before release.

> An agent can help when the next action depends on what it finds. Missing documents may require another retrieval step, while conflicting records may require escalation. I would bound the workflow with explicit tool permissions, review requirements, and stopping conditions.

WHO identifies healthcare documentation and administrative assistance among potential applications. Its guidance also discusses inaccurate outputs, bias, privacy risks, and automation bias. [WHO guidance](https://www.who.int/news/item/18-01-2024-who-releases-ai-ethics-and-governance-guidance-for-large-multi-modal-models)

Adapting NeuroAgent would require clinically labeled validation, appropriate data integrations, workflow-specific evaluation, and governance. General research benchmark scores do not establish healthcare readiness.

## Which healthcare evaluation parameters matter?

Define acceptance criteria with clinicians, workflow owners, and privacy and security specialists. The metrics and thresholds should reflect the consequences of errors in the specific task.

| Area | What to evaluate |
| --- | --- |
| Patient and encounter matching | Whether each fact belongs to the intended patient and visit |
| Critical errors and omissions | Incorrect medication details, missed allergies, or missing relevant information |
| Evidence quality | Supporting source passages, dates, document versions, and applicability |
| Uncertainty handling | Appropriate escalation when evidence is missing or contradictory |
| Performance across groups | Error differences across relevant populations, languages, and sites |
| Workflow usefulness | Review time, correction rate, and proportion of drafts accepted |
| Reliability | Tool failures, duplicate actions, recovery behavior, and audit completeness |
| Cost and latency | Cost per accepted case and response-time distributions, including slower cases |

For classification, sensitivity, specificity, and positive predictive value may be relevant. For extraction, field correctness and clinically significant omissions may be more useful. An overall accuracy number can hide consequential errors. Model-based critique is one signal; independent domain evaluation remains necessary.

For a HIPAA-regulated U.S. workflow, assess every service handling protected information, including the model API, retrieval store, observability platform, and sandbox. Applicable business associate agreements and safeguards form part of that assessment. NeuroAgent's current third-party integrations would each require review for the intended healthcare configuration. [HHS cloud guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/health-information-technology/cloud-computing/index.html)

Clinical decision-support functionality needs regulatory assessment based on intended use and applicable FDA criteria. Having a human reviewer alone does not determine its regulatory status. [FDA clinical decision support guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)

## How would you approach an agentic AI problem efficiently?

1. **Define success.** Identify the user, desired outcome, current baseline, acceptable errors, budget, and response-time requirement.
2. **Inspect the workflow and data.** Understand APIs, document quality, access permissions, and operational constraints.
3. **Choose a suitable architecture.** Use rules for fixed decisions, retrieval for evidence lookup, and an agent when observations determine subsequent actions.
4. **Build an evaluation set early.** Cover normal cases, missing information, conflicting sources, and tool failures. Keep development and final evaluation examples separate, with leakage-aware splits appropriate to the data.
5. **Build a narrow version.** Limit tools and permissions. Define structured outputs, evidence requirements, stopping conditions, and spending limits.
6. **Evaluate the whole workflow.** Measure quality, critical errors, review effort, latency, and total cost against the baseline.
7. **Pilot and monitor.** Begin in shadow mode, where outputs are evaluated without operational actions. Introduce reviewed use with clear owners, version tracking, monitoring, and rollback.

Interviewers can assess problem framing, justified architecture choices, measurement, communication, and operational ownership through this approach.

## Short interview answer to rehearse

> I would start with the business workflow and its failure costs, establish a baseline, and build the smallest system that can improve it. My NeuroAgent experiments taught me to measure whether extra reasoning and retries earn their cost. I would compare models on representative tasks and optimize cost per accepted outcome. In healthcare, I would define evidence requirements, critical-error thresholds, privacy controls, and human review with domain experts before expanding the agent's responsibilities.
