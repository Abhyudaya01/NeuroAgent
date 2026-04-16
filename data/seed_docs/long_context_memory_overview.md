# Long-Context Memory in LLM Agents

Long-context memory research in LLM agents focuses on preserving relevant information across long interactions without forcing the base model to keep every token inside a single prompt window.

Common approaches include retrieval-augmented memory, recurrent summarization, memory compression, and external key-value memory stores. Retrieval-augmented systems store observations and later fetch only the most relevant chunks for the current query. Recurrent summarization condenses previous conversation or environment state into compact summaries. Compression methods reduce long histories into smaller latent or textual representations.

For agent workflows, long-context memory matters because the agent must maintain task goals, previous tool outputs, intermediate findings, and user preferences across many steps. Weak memory leads to repeated searches, inconsistent reasoning, and loss of important evidence.
