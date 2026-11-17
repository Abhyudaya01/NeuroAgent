"""Conservative GPT-4o chat estimates using published per-million token rates.

Rates checked 2026-11-17 at https://developers.openai.com/api/docs/models/gpt-4o.
Cached inputs are charged at the full input rate in these estimates. External
tools, embeddings, and unrelated account activity are outside this calculation.
"""

GPT4O_PRICES = {"input_per_million": 2.50, "output_per_million": 10.00}


def estimate_chat_cost(usage_by_model: dict) -> float:
    cost = 0.0
    for model, usage in usage_by_model.items():
        if model != "gpt-4o" and not model.startswith("gpt-4o-2024-"):
            raise ValueError(f"No verified price configured for model {model!r}.")
        cost += (
            usage.get("input_tokens", 0) * GPT4O_PRICES["input_per_million"]
            + usage.get("output_tokens", 0) * GPT4O_PRICES["output_per_million"]
        ) / 1_000_000
    return cost
