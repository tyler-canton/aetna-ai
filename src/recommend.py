"""
Movie recommendation system with LLM integration.

Supports:
- Natural language queries ("action movies with high ratings")
- Personalized recommendations based on user history
- Comparative analysis between movies
- Preference summarization

Usage:
    python -m src.recommend "action movies with high ratings"
    python -m src.recommend --user 42
    python -m src.recommend --compare "The Dark Knight" "Spider-Man"
    python -m src.recommend --summarize --user 42
"""

import argparse
import json
import sys
from typing import Any

from pydantic import ValidationError

from .db import (
    get_movie_by_id,
    get_user_rated_movies,
    get_top_rated_movies,
    search_movies,
    search_movies_by_title,
    get_enriched_movie,
)
from .models import (
    SearchFilters,
    MovieRecommendation,
    UserPreferenceSummary,
    MovieComparison,
)
from .prompts import (
    QUERY_PARSER_SYSTEM_PROMPT,
    QUERY_PARSER_USER_PROMPT,
    RECOMMENDATION_SYSTEM_PROMPT,
    RECOMMENDATION_USER_PROMPT,
    PREFERENCE_SUMMARY_SYSTEM_PROMPT,
    PREFERENCE_SUMMARY_USER_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_USER_PROMPT,
)
from .utils import (
    get_openai_client,
    get_model_name,
    format_movie_for_display,
    format_enrichment_for_display,
    validate_api_key,
    extract_year_from_date,
)


def parse_query(query: str) -> SearchFilters:
    """
    Use LLM to parse natural language query into search filters.

    Args:
        query: Natural language search query

    Returns:
        SearchFilters with extracted parameters
    """
    client = get_openai_client()
    model = get_model_name()

    user_prompt = QUERY_PARSER_USER_PROMPT.format(query=query)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        content = response.choices[0].message.content
        if not content:
            return SearchFilters()

        data = json.loads(content)
        return SearchFilters(**data)

    except (json.JSONDecodeError, ValidationError) as e:
        print(f"⚠️  Could not parse query: {e}")
        return SearchFilters()


def explain_recommendation(
    movie: dict[str, Any],
    request: str,
    enrichment: dict[str, Any] | None = None
) -> MovieRecommendation:
    """
    Use LLM to explain why a movie is recommended.

    Args:
        movie: Movie data dict
        request: Original user request
        enrichment: Optional enrichment data

    Returns:
        MovieRecommendation with explanation and match score
    """
    client = get_openai_client()
    model = get_model_name()

    # Format genres
    genres = movie.get("genres", [])
    if isinstance(genres, list):
        genres_str = ", ".join(genres) if genres else "Unknown"
    else:
        genres_str = str(genres)

    # Format enrichment info if available
    enrichment_info = ""
    if enrichment:
        enrichment_info = f"""
- Sentiment: {enrichment.get('sentiment_score', 'N/A')}
- Themes: {', '.join(enrichment.get('themes', [])) if enrichment.get('themes') else 'N/A'}
- Budget Tier: {enrichment.get('budget_tier', 'N/A')}
- ROI: {enrichment.get('roi_category', 'N/A')}"""

    year = extract_year_from_date(movie.get("releaseDate")) or "Unknown"

    user_prompt = RECOMMENDATION_USER_PROMPT.format(
        request=request,
        title=movie.get("title", "Unknown"),
        overview=movie.get("overview", "No overview available"),
        genres=genres_str,
        year=year,
        enrichment_info=enrichment_info,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        content = response.choices[0].message.content
        if not content:
            return MovieRecommendation(
                movie=movie,
                explanation="This movie matches your search criteria.",
                match_score=0.5,
            )

        data = json.loads(content)
        return MovieRecommendation(
            movie=movie,
            explanation=data.get("explanation", "Matches your criteria."),
            match_score=data.get("match_score", 0.5),
        )

    except Exception as e:
        print(f"⚠️  Could not generate explanation: {e}")
        return MovieRecommendation(
            movie=movie,
            explanation="This movie matches your search criteria.",
            match_score=0.5,
        )


def recommend_from_query(query: str, limit: int = 10) -> list[MovieRecommendation]:
    """
    Get movie recommendations based on natural language query.

    Args:
        query: Natural language search query
        limit: Maximum number of recommendations

    Returns:
        List of MovieRecommendation objects
    """
    print(f"\n🔍 Parsing query: \"{query}\"")

    # Parse the query
    filters = parse_query(query)
    print(f"   Extracted filters: {filters.model_dump(exclude_none=True)}")

    # Search for movies
    movies = search_movies(
        genres=filters.genres,
        min_year=filters.min_year,
        max_year=filters.max_year,
        min_rating=filters.min_rating,
        limit=limit * 2,  # Get more to filter
    )

    if not movies:
        print("   No movies found matching criteria")
        return []

    print(f"   Found {len(movies)} potential matches")

    # Generate explanations for top results
    recommendations = []
    for movie in movies[:limit]:
        enrichment = get_enriched_movie(movie["movieId"])
        rec = explain_recommendation(movie, query, enrichment)
        recommendations.append(rec)

    # Sort by match score
    recommendations.sort(key=lambda r: r.match_score, reverse=True)

    return recommendations


def recommend_for_user(user_id: int, limit: int = 10) -> list[MovieRecommendation]:
    """
    Get personalized recommendations based on user's rating history.

    Args:
        user_id: User ID
        limit: Maximum number of recommendations

    Returns:
        List of MovieRecommendation objects
    """
    print(f"\n👤 Getting recommendations for user {user_id}")

    # Get user's rated movies
    rated_movies = get_user_rated_movies(user_id)
    if not rated_movies:
        print("   User has no ratings")
        return []

    print(f"   Found {len(rated_movies)} rated movies")

    # Analyze preferences from highly rated movies
    high_rated = [m for m in rated_movies if m.get("rating", 0) >= 4.0]
    if not high_rated:
        high_rated = rated_movies[:5]

    # Extract preferred genres
    genre_counts: dict[str, int] = {}
    for movie in high_rated:
        for genre in movie.get("genres", []):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    top_genres = sorted(genre_counts.keys(), key=lambda g: genre_counts[g], reverse=True)[:3]
    print(f"   Preferred genres: {top_genres}")

    # Get movies user hasn't rated
    rated_ids = {m["movieId"] for m in rated_movies}

    # Search for similar movies
    candidates = search_movies(genres=top_genres, min_rating=3.5, limit=limit * 3)
    candidates = [m for m in candidates if m["movieId"] not in rated_ids]

    if not candidates:
        print("   No new recommendations found")
        return []

    # Generate explanations
    request = f"Movies similar to user's favorites in genres: {', '.join(top_genres)}"
    recommendations = []

    for movie in candidates[:limit]:
        enrichment = get_enriched_movie(movie["movieId"])
        rec = explain_recommendation(movie, request, enrichment)
        recommendations.append(rec)

    recommendations.sort(key=lambda r: r.match_score, reverse=True)

    return recommendations


def summarize_user_preferences(user_id: int) -> UserPreferenceSummary | None:
    """
    Generate a summary of user's movie preferences.

    Args:
        user_id: User ID

    Returns:
        UserPreferenceSummary or None if user has no ratings
    """
    print(f"\n📊 Analyzing preferences for user {user_id}")

    rated_movies = get_user_rated_movies(user_id)
    if not rated_movies:
        print("   User has no ratings")
        return None

    # Calculate stats
    total_ratings = len(rated_movies)
    avg_rating = sum(m.get("rating", 0) for m in rated_movies) / total_ratings

    high_rated = [m for m in rated_movies if m.get("rating", 0) >= 4.0]
    low_rated = [m for m in rated_movies if m.get("rating", 0) <= 2.0]

    # Format for prompt
    def format_movie_brief(m: dict) -> str:
        genres = ", ".join(m.get("genres", [])[:3]) if m.get("genres") else "Unknown"
        return f"- {m.get('title')} ({m.get('rating')}/5) [{genres}]"

    high_rated_str = "\n".join(format_movie_brief(m) for m in high_rated[:10])
    low_rated_str = "\n".join(format_movie_brief(m) for m in low_rated[:5])

    if not high_rated_str:
        high_rated_str = "No movies rated 4+ stars"
    if not low_rated_str:
        low_rated_str = "No movies rated 2 stars or below"

    client = get_openai_client()
    model = get_model_name()

    user_prompt = PREFERENCE_SUMMARY_USER_PROMPT.format(
        user_id=user_id,
        total_ratings=total_ratings,
        avg_rating=avg_rating,
        high_rated_movies=high_rated_str,
        low_rated_movies=low_rated_str,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PREFERENCE_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        content = response.choices[0].message.content
        if not content:
            return None

        data = json.loads(content)
        data["user_id"] = user_id
        return UserPreferenceSummary(**data)

    except Exception as e:
        print(f"⚠️  Could not generate summary: {e}")
        return None


def compare_movies(title1: str, title2: str) -> MovieComparison | None:
    """
    Compare two movies and provide analysis.

    Args:
        title1: First movie title
        title2: Second movie title

    Returns:
        MovieComparison or None if movies not found
    """
    print(f"\n🎬 Comparing: \"{title1}\" vs \"{title2}\"")

    # Find movies
    movies1 = search_movies_by_title(title1, limit=1)
    movies2 = search_movies_by_title(title2, limit=1)

    if not movies1:
        print(f"   Could not find: {title1}")
        return None
    if not movies2:
        print(f"   Could not find: {title2}")
        return None

    movie1 = movies1[0]
    movie2 = movies2[0]

    # Format genres
    def format_genres(m: dict) -> str:
        genres = m.get("genres", [])
        return ", ".join(genres) if genres else "Unknown"

    client = get_openai_client()
    model = get_model_name()

    user_prompt = COMPARISON_USER_PROMPT.format(
        title1=movie1.get("title"),
        overview1=movie1.get("overview", "No overview"),
        genres1=format_genres(movie1),
        year1=extract_year_from_date(movie1.get("releaseDate")) or "Unknown",
        budget1=movie1.get("budget", 0),
        revenue1=movie1.get("revenue", 0),
        title2=movie2.get("title"),
        overview2=movie2.get("overview", "No overview"),
        genres2=format_genres(movie2),
        year2=extract_year_from_date(movie2.get("releaseDate")) or "Unknown",
        budget2=movie2.get("budget", 0),
        revenue2=movie2.get("revenue", 0),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        content = response.choices[0].message.content
        if not content:
            return None

        data = json.loads(content)
        data["movie1_title"] = movie1.get("title")
        data["movie2_title"] = movie2.get("title")
        return MovieComparison(**data)

    except Exception as e:
        print(f"⚠️  Could not generate comparison: {e}")
        return None


def display_recommendations(recommendations: list[MovieRecommendation]) -> None:
    """Display recommendations in a readable format."""
    print(f"\n🎬 Recommendations ({len(recommendations)} movies):\n")

    for i, rec in enumerate(recommendations, 1):
        movie = rec.movie if hasattr(rec.movie, "model_dump") else rec.movie
        if hasattr(movie, "model_dump"):
            movie = movie.model_dump()

        print(f"{i}. {movie.get('title', 'Unknown')}")

        if movie.get("releaseDate"):
            print(f"   Year: {movie['releaseDate'][:4]}")

        if movie.get("genres"):
            genres = movie["genres"]
            if isinstance(genres, list):
                genres = ", ".join(genres)
            print(f"   Genres: {genres}")

        print(f"   Match Score: {'⭐' * int(rec.match_score * 5)} ({rec.match_score:.0%})")
        print(f"   💡 {rec.explanation}")
        print()


def display_preference_summary(summary: UserPreferenceSummary) -> None:
    """Display user preference summary."""
    print(f"\n📊 Preference Summary for User {summary.user_id}\n")
    print(f"Favorite Genres: {', '.join(summary.favorite_genres)}")
    print(f"Preferred Themes: {', '.join(summary.preferred_themes)}")
    print(f"Rating Tendency: {summary.rating_tendency}")
    print(f"\n{summary.summary}")


def display_comparison(comparison: MovieComparison) -> None:
    """Display movie comparison."""
    print(f"\n🎬 {comparison.movie1_title} vs {comparison.movie2_title}\n")

    print("Similarities:")
    for sim in comparison.similarities:
        print(f"  • {sim}")

    print("\nDifferences:")
    for diff in comparison.differences:
        print(f"  • {diff}")

    print(f"\nRecommendation:\n  {comparison.recommendation}")


def main():
    """CLI entry point for recommendation system."""
    parser = argparse.ArgumentParser(
        description="Movie recommendation system with LLM"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language query for recommendations"
    )
    parser.add_argument(
        "--user",
        type=int,
        help="User ID for personalized recommendations"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("MOVIE1", "MOVIE2"),
        help="Compare two movies"
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize user preferences (requires --user)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of recommendations (default: 10)"
    )

    args = parser.parse_args()

    # Validate API key
    if not validate_api_key():
        print("❌ Error: OPENAI_API_KEY not set or invalid")
        print("   Set it in your environment or .env file")
        sys.exit(1)

    print("🎬 Movie Recommendation System")
    print("=" * 40)

    # Handle different modes
    if args.compare:
        comparison = compare_movies(args.compare[0], args.compare[1])
        if comparison:
            display_comparison(comparison)
        else:
            print("Could not compare movies")

    elif args.summarize:
        if not args.user:
            print("❌ Error: --summarize requires --user")
            sys.exit(1)
        summary = summarize_user_preferences(args.user)
        if summary:
            display_preference_summary(summary)
        else:
            print("Could not generate preference summary")

    elif args.user:
        recommendations = recommend_for_user(args.user, limit=args.limit)
        if recommendations:
            display_recommendations(recommendations)
        else:
            print("No recommendations found")

    elif args.query:
        recommendations = recommend_from_query(args.query, limit=args.limit)
        if recommendations:
            display_recommendations(recommendations)
        else:
            print("No recommendations found")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
