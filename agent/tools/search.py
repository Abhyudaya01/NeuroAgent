"""Web search tool backed by the Tavily Python client."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

from dotenv import load_dotenv
from tavily import TavilyClient

from agent.tools import ToolResult


MIN_FINDING_CHARS = 300
"""Minimum usable finding length for a retained web-search result."""


BLOCKED_DOMAINS = {
    "linkedin.com",
    "www.linkedin.com",
    "twitter.com",
    "www.twitter.com",
    "x.com",
    "www.x.com",
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "tiktok.com",
    "www.tiktok.com",
}
"""Domains excluded from search results because they are weak research sources."""

HIGH_TRUST_DOMAINS = {
    "arxiv.org",
    "www.arxiv.org",
    "acm.org",
    "www.acm.org",
    "ieee.org",
    "www.ieee.org",
    "nature.com",
    "www.nature.com",
    "semanticscholar.org",
    "www.semanticscholar.org",
}
"""Domains that start at the highest web-search confidence tier."""

OTHER_ACADEMIC_DOMAINS = {
    "openreview.net",
    "www.openreview.net",
    "science.org",
    "www.science.org",
    "aclweb.org",
    "www.aclweb.org",
    "proceedings.neurips.cc",
    "jmlr.org",
    "www.jmlr.org",
}
"""Academic or conference domains that receive the middle confidence tier."""

NEWS_BLOG_DOMAINS = {
    "developer.nvidia.com",
    "openai.com",
    "www.openai.com",
    "blogs.microsoft.com",
    "huggingface.co",
    "www.huggingface.co",
    "techcrunch.com",
    "www.techcrunch.com",
}
"""News and blog style domains that receive the lower but still usable tier."""


def _domain(url: str) -> str:
    """Extract the hostname from a URL."""

    return urlparse(url).netloc.lower()


def _is_allowed_result(result: dict, min_chars: int = MIN_FINDING_CHARS) -> bool:
    """Return whether a Tavily result should be kept for research use."""

    url = result.get("url", "")
    if not url:
        return False
    if _domain(url) in BLOCKED_DOMAINS:
        return False
    return len(_best_finding_text(result)) >= min_chars


def _best_finding_text(result: dict) -> str:
    """Return the best available full finding text from Tavily."""

    content = (result.get("content") or "").strip()
    if content:
        return content
    return (result.get("raw_content") or "").strip()


def _result_weight(result: dict) -> float:
    """Compute a deterministic confidence score for a filtered search result."""

    hostname = _domain(result.get("url", ""))
    content = _best_finding_text(result)
    title = result.get("title", "") or ""
    combined_text = f"{title} {content}"
    score = _domain_trust_score(hostname)

    if re.search(r'"[^"]{8,}"', combined_text):
        score += 0.05

    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", combined_text):
        score += 0.05

    if re.search(r"\b202[67]\b", combined_text):
        score -= 0.10

    return max(0.0, min(0.85, score))


def _domain_trust_score(hostname: str) -> float:
    """Return the base trust tier for a search result domain."""

    hostname = hostname.lower()
    if hostname in HIGH_TRUST_DOMAINS:
        return 0.80
    if hostname in OTHER_ACADEMIC_DOMAINS or hostname.endswith(".edu"):
        return 0.70
    if hostname in NEWS_BLOG_DOMAINS:
        return 0.55
    return 0.50


def _format_result(result: dict) -> str:
    """Render a Tavily result into a structured evidence block."""

    title = result.get("title", "Untitled source").strip()
    url = result.get("url", "").strip()
    content = _best_finding_text(result)
    return f"## {title}\nURL: {url}\nKey finding: {content.strip()}\n"


def run_web_search(query: str, *, fetch_sources: bool = False) -> ToolResult:
    """Run a Tavily web search for the supplied query.

    On success, the returned content includes structured blocks for the
    top search results. On failure, an empty result with zero
    confidence is returned instead of raising.
    """

    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return ToolResult(content="", sources=[], raw_confidence=0.0, tool_name="web_search")

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )
        results = response.get("results", [])
        filtered_results = [
            result for result in results
            if _is_allowed_result(result, min_chars=1 if fetch_sources else MIN_FINDING_CHARS)
        ]
        ranked_results = sorted(
            filtered_results,
            key=_result_weight,
            reverse=True,
        )[:3]

        if fetch_sources and ranked_results:
            urls = [result["url"] for result in ranked_results[:2] if result.get("url")]
            try:
                extracted = client.extract(
                    urls=urls, query=query, chunks_per_source=3,
                    format="markdown", extract_depth="basic", timeout=15,
                )
                passages = {item["url"]: item.get("raw_content", "") for item in extracted.get("results", [])}
                for result in ranked_results:
                    if passages.get(result.get("url")):
                        result["content"] = _best_finding_text(result) + "\n\n" + passages[result["url"]]
            except Exception:
                # Keep search evidence available if page extraction fails.
                pass
        formatted_results = [_format_result(result) for result in ranked_results]
        sources = [result.get("url", "") for result in ranked_results if result.get("url")]
        trust_scores = [_domain_trust_score(_domain(result.get("url", ""))) for result in ranked_results]
        raw_confidence = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0

        return ToolResult(
            content="\n".join(formatted_results).strip(),
            sources=sources,
            source_content={result.get("url", ""): _format_result(result) for result in ranked_results if result.get("url")},
            raw_confidence=raw_confidence,
            tool_name="web_search",
        )
    except Exception:
        return ToolResult(content="", sources=[], raw_confidence=0.0, tool_name="web_search")
