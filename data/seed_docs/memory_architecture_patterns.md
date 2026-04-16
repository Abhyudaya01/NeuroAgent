# Memory Architecture Patterns

Several memory architecture patterns appear repeatedly in recent LLM-agent research:

1. Episodic memory stores prior experiences or observations and retrieves them using semantic similarity.
2. Semantic memory stores stable facts, extracted knowledge, or learned summaries that can be reused across tasks.
3. Working memory keeps a small, task-specific context window active for immediate reasoning.
4. Hierarchical memory combines short-term buffers with longer-term compressed summaries or vector indexes.

Promising systems often combine retrieval with summarization. Retrieval preserves detail, while summarization controls context growth. Hybrid memory systems are especially useful when the agent must balance precision, cost, and context length limits.
