"""
Shared utilities for the movie recommendation system.
"""

import asyncio
import os
from functools import lru_cache
from typing import TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

# Load environment variables
load_dotenv()

T = TypeVar("T")


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Get synchronous OpenAI client (cached)."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def get_async_openai_client() -> AsyncOpenAI:
    """Get async OpenAI client (cached)."""
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_model_name() -> str:
    """Get the model name to use for LLM calls."""
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


async def batch_process(
    items: list[T],
    processor,
    batch_size: int = 10,
    delay_between_batches: float = 1.0
) -> list:
    """
    Process items in batches with delay between batches.

    Args:
        items: List of items to process
        processor: Async function to process each item
        batch_size: Number of items per batch
        delay_between_batches: Seconds to wait between batches

    Returns:
        List of results from processor
    """
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        # Process batch concurrently
        batch_results = await asyncio.gather(
            *[processor(item) for item in batch],
            return_exceptions=True
        )

        results.extend(batch_results)

        # Delay between batches (except for last batch)
        if i + batch_size < len(items):
            await asyncio.sleep(delay_between_batches)

    return results


def format_movie_for_display(movie: dict) -> str:
    """Format a movie dict for human-readable display."""
    lines = [f"🎬 {movie.get('title', 'Unknown')}"]

    if movie.get("releaseDate"):
        year = movie["releaseDate"][:4]
        lines.append(f"   Year: {year}")

    if movie.get("genres"):
        genres = movie["genres"]
        if isinstance(genres, list):
            genres = ", ".join(genres)
        lines.append(f"   Genres: {genres}")

    if movie.get("overview"):
        overview = movie["overview"]
        if len(overview) > 200:
            overview = overview[:200] + "..."
        lines.append(f"   {overview}")

    if movie.get("avg_rating"):
        lines.append(f"   ⭐ {movie['avg_rating']:.1f}/5 ({movie.get('num_ratings', 0)} ratings)")

    if movie.get("budget") and movie.get("revenue"):
        budget = movie["budget"]
        revenue = movie["revenue"]
        roi = revenue / budget if budget > 0 else 0
        lines.append(f"   💰 Budget: ${budget:,} | Revenue: ${revenue:,} | ROI: {roi:.1f}x")

    return "\n".join(lines)


def format_enrichment_for_display(enriched: dict) -> str:
    """Format enriched movie data for display."""
    lines = []

    if enriched.get("sentiment_score") is not None:
        score = enriched["sentiment_score"]
        emoji = "😊" if score > 0.3 else "😐" if score > -0.3 else "😢"
        lines.append(f"   Sentiment: {emoji} {score:.2f}")

    if enriched.get("budget_tier"):
        lines.append(f"   Budget Tier: {enriched['budget_tier']}")

    if enriched.get("revenue_tier"):
        lines.append(f"   Revenue Tier: {enriched['revenue_tier']}")

    if enriched.get("roi_category"):
        lines.append(f"   ROI Category: {enriched['roi_category']}")

    if enriched.get("themes"):
        themes = enriched["themes"]
        if isinstance(themes, list):
            themes = ", ".join(themes)
        lines.append(f"   Themes: {themes}")

    return "\n".join(lines)


def validate_api_key() -> bool:
    """Check if OpenAI API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    return bool(api_key and api_key.startswith("sk-"))


def extract_year_from_date(date_str: str | None) -> int | None:
    """Extract year from a YYYY-MM-DD date string."""
    if not date_str or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except ValueError:
        return None
