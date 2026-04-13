"""
Data enrichment pipeline for movies using LLM.

Generates 5 attributes for each movie:
- sentiment_score: Sentiment analysis of overview (-1.0 to 1.0)
- budget_tier: micro/low/medium/high/blockbuster
- revenue_tier: flop/underperformer/moderate/hit/blockbuster
- roi_category: disaster/loss/break_even/profitable/smash_hit
- themes: 3-5 thematic keywords

Usage:
    python -m src.enrich --limit 100
"""

import argparse
import asyncio
import json
import sys
from typing import Any

from pydantic import ValidationError

from .db import get_movies_for_enrichment, save_enriched_movie, get_enriched_movie
from .models import EnrichedAttributes
from .prompts import ENRICHMENT_SYSTEM_PROMPT, ENRICHMENT_USER_PROMPT
from .utils import (
    get_async_openai_client,
    get_model_name,
    batch_process,
    format_movie_for_display,
    format_enrichment_for_display,
    validate_api_key,
)


async def enrich_single_movie(movie: dict[str, Any]) -> dict[str, Any] | None:
    """
    Enrich a single movie with LLM-generated attributes.

    Args:
        movie: Movie dict with title, overview, budget, revenue, genres

    Returns:
        Enriched attributes dict or None if failed
    """
    client = get_async_openai_client()
    model = get_model_name()

    # Format genres for prompt
    genres = movie.get("genres", [])
    if isinstance(genres, list):
        genres_str = ", ".join(genres) if genres else "Unknown"
    else:
        genres_str = str(genres)

    # Build the user prompt
    user_prompt = ENRICHMENT_USER_PROMPT.format(
        title=movie.get("title", "Unknown"),
        overview=movie.get("overview", "No overview available"),
        budget=movie.get("budget", 0),
        revenue=movie.get("revenue", 0),
        genres=genres_str,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,  # Lower temperature for more consistent outputs
        )

        # Parse the response
        content = response.choices[0].message.content
        if not content:
            print(f"  ⚠️  Empty response for {movie.get('title')}")
            return None

        data = json.loads(content)

        # Validate with Pydantic
        enriched = EnrichedAttributes(**data)

        return {
            "movieId": movie["movieId"],
            "sentiment_score": enriched.sentiment_score,
            "budget_tier": enriched.budget_tier,
            "revenue_tier": enriched.revenue_tier,
            "roi_category": enriched.roi_category,
            "themes": enriched.themes,
            "model_used": model,
        }

    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error for {movie.get('title')}: {e}")
        return None
    except ValidationError as e:
        print(f"  ⚠️  Validation error for {movie.get('title')}: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️  Error enriching {movie.get('title')}: {e}")
        return None


async def enrich_movies(
    limit: int = 100,
    batch_size: int = 10,
    skip_existing: bool = True
) -> list[dict[str, Any]]:
    """
    Enrich multiple movies with LLM-generated attributes.

    Args:
        limit: Maximum number of movies to enrich
        batch_size: Number of movies to process in parallel
        skip_existing: Skip movies that already have enrichment data

    Returns:
        List of enriched movie data
    """
    print(f"\n📊 Fetching movies for enrichment (limit: {limit})...")
    movies = get_movies_for_enrichment(limit)
    print(f"   Found {len(movies)} movies with complete data")

    # Filter out already enriched movies if requested
    if skip_existing:
        movies_to_process = []
        for movie in movies:
            existing = get_enriched_movie(movie["movieId"])
            if existing is None:
                movies_to_process.append(movie)

        skipped = len(movies) - len(movies_to_process)
        if skipped > 0:
            print(f"   Skipping {skipped} already enriched movies")
        movies = movies_to_process

    if not movies:
        print("   No movies to enrich!")
        return []

    print(f"\n🤖 Starting enrichment of {len(movies)} movies...")
    print(f"   Batch size: {batch_size}")
    print(f"   Model: {get_model_name()}")

    results = []
    failed = 0

    for i in range(0, len(movies), batch_size):
        batch = movies[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(movies) + batch_size - 1) // batch_size

        print(f"\n   Batch {batch_num}/{total_batches}...")

        # Process batch concurrently
        batch_results = await asyncio.gather(
            *[enrich_single_movie(movie) for movie in batch],
            return_exceptions=True
        )

        # Save successful results
        for j, result in enumerate(batch_results):
            movie = batch[j]
            if isinstance(result, Exception):
                print(f"     ❌ {movie.get('title')}: {result}")
                failed += 1
            elif result is None:
                failed += 1
            else:
                # Save to database
                save_enriched_movie(
                    movie_id=result["movieId"],
                    sentiment_score=result["sentiment_score"],
                    budget_tier=result["budget_tier"],
                    revenue_tier=result["revenue_tier"],
                    roi_category=result["roi_category"],
                    themes=result["themes"],
                    model_used=result["model_used"],
                )
                results.append(result)
                print(f"     ✓ {movie.get('title')}")

        # Small delay between batches to avoid rate limits
        if i + batch_size < len(movies):
            await asyncio.sleep(1.0)

    print(f"\n✅ Enrichment complete!")
    print(f"   Successful: {len(results)}")
    print(f"   Failed: {failed}")

    return results


def display_enriched_movies(movies: list[dict[str, Any]], limit: int = 10) -> None:
    """Display enriched movies in a readable format."""
    print(f"\n📋 Sample of enriched movies (showing {min(limit, len(movies))}):\n")

    for movie in movies[:limit]:
        # Get movie details
        from .db import get_movie_by_id
        details = get_movie_by_id(movie["movieId"])

        if details:
            print(format_movie_for_display(details))
        else:
            print(f"🎬 Movie ID: {movie['movieId']}")

        print(format_enrichment_for_display(movie))
        print()


def main():
    """CLI entry point for enrichment pipeline."""
    parser = argparse.ArgumentParser(
        description="Enrich movies with LLM-generated attributes"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of movies to enrich (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for parallel processing (default: 10)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-enrich movies that already have data"
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="Number of results to display (default: 5)"
    )

    args = parser.parse_args()

    # Validate API key
    if not validate_api_key():
        print("❌ Error: OPENAI_API_KEY not set or invalid")
        print("   Set it in your environment or .env file")
        sys.exit(1)

    print("🎬 Movie Enrichment Pipeline")
    print("=" * 40)

    # Run enrichment
    results = asyncio.run(
        enrich_movies(
            limit=args.limit,
            batch_size=args.batch_size,
            skip_existing=not args.force,
        )
    )

    # Display results
    if results and args.show > 0:
        display_enriched_movies(results, limit=args.show)


if __name__ == "__main__":
    main()
