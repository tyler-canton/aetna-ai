"""
Database connections and query functions for movies and ratings.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

# Database paths
DB_DIR = Path(__file__).parent.parent / "db"
MOVIES_DB = DB_DIR / "movies.db"
RATINGS_DB = DB_DIR / "ratings.db"
ENRICHED_DB = DB_DIR / "enriched_movies.db"


def _parse_genres(genres_json: str | None) -> list[str]:
    """Parse genres JSON, handling both dict and string formats."""
    if not genres_json:
        return []
    try:
        genres_data = json.loads(genres_json)
        if not genres_data:
            return []
        # Handle [{"id": 28, "name": "Action"}] format
        if isinstance(genres_data[0], dict):
            return [g.get("name", "") for g in genres_data if g.get("name")]
        # Handle ["Action", "Drama"] format
        return genres_data
    except (json.JSONDecodeError, IndexError, TypeError):
        return []


def get_movies_connection() -> sqlite3.Connection:
    """Get connection to movies database."""
    conn = sqlite3.connect(MOVIES_DB)
    conn.row_factory = sqlite3.Row
    return conn


def get_ratings_connection() -> sqlite3.Connection:
    """Get connection to ratings database."""
    conn = sqlite3.connect(RATINGS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def get_enriched_connection() -> sqlite3.Connection:
    """Get connection to enriched movies database, creating if needed."""
    conn = sqlite3.connect(ENRICHED_DB)
    conn.row_factory = sqlite3.Row
    _init_enriched_table(conn)
    return conn


def _init_enriched_table(conn: sqlite3.Connection) -> None:
    """Create enriched_movies table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS enriched_movies (
            movieId INTEGER PRIMARY KEY,
            sentiment_score REAL,
            budget_tier TEXT,
            revenue_tier TEXT,
            roi_category TEXT,
            themes TEXT,
            enriched_at TEXT,
            model_used TEXT
        )
    """)
    conn.commit()


def get_movies_for_enrichment(limit: int = 100) -> list[dict[str, Any]]:
    """
    Get movies suitable for LLM enrichment.

    Selects movies with:
    - Non-null overview (>100 chars)
    - Budget > 0 and Revenue > 0
    - Ordered by revenue DESC
    """
    conn = get_movies_connection()
    cursor = conn.execute("""
        SELECT movieId, title, overview, budget, revenue, genres,
               releaseDate, runtime, language
        FROM movies
        WHERE overview IS NOT NULL
          AND LENGTH(overview) > 100
          AND budget > 0
          AND revenue > 0
        ORDER BY revenue DESC
        LIMIT ?
    """, (limit,))

    movies = []
    for row in cursor.fetchall():
        movie = dict(row)
        movie["genres"] = _parse_genres(movie.get("genres"))
        movies.append(movie)

    conn.close()
    return movies


def get_movie_by_id(movie_id: int) -> dict[str, Any] | None:
    """Get a single movie by ID."""
    conn = get_movies_connection()
    cursor = conn.execute("""
        SELECT movieId, title, overview, budget, revenue, genres,
               releaseDate, runtime, language, imdbId, status,
               productionCompanies
        FROM movies
        WHERE movieId = ?
    """, (movie_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    movie = dict(row)
    # Parse JSON fields
    movie["genres"] = _parse_genres(movie.get("genres"))
    # productionCompanies also uses {"name": ...} format
    if movie.get("productionCompanies"):
        try:
            companies = json.loads(movie["productionCompanies"])
            if companies and isinstance(companies[0], dict):
                movie["productionCompanies"] = [c.get("name", "") for c in companies if c.get("name")]
            else:
                movie["productionCompanies"] = companies
        except (json.JSONDecodeError, IndexError, TypeError):
            movie["productionCompanies"] = []
    else:
        movie["productionCompanies"] = []

    return movie


def search_movies_by_title(title: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search movies by title (case-insensitive partial match)."""
    conn = get_movies_connection()
    cursor = conn.execute("""
        SELECT movieId, title, overview, budget, revenue, genres,
               releaseDate, runtime, language
        FROM movies
        WHERE title LIKE ?
        ORDER BY revenue DESC
        LIMIT ?
    """, (f"%{title}%", limit))

    movies = [dict(row) for row in cursor.fetchall()]
    conn.close()

    for movie in movies:
        movie["genres"] = _parse_genres(movie.get("genres"))

    return movies


def get_user_ratings(user_id: int) -> list[dict[str, Any]]:
    """Get all ratings for a specific user."""
    conn = get_ratings_connection()
    cursor = conn.execute("""
        SELECT ratingId, userId, movieId, rating, timestamp
        FROM ratings
        WHERE userId = ?
        ORDER BY rating DESC
    """, (user_id,))

    ratings = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return ratings


def get_user_rated_movies(user_id: int) -> list[dict[str, Any]]:
    """Get movies that a user has rated, with their ratings."""
    movies_conn = get_movies_connection()

    # Attach ratings database
    movies_conn.execute(f"ATTACH DATABASE '{RATINGS_DB}' AS r")

    cursor = movies_conn.execute("""
        SELECT m.movieId, m.title, m.overview, m.budget, m.revenue,
               m.genres, m.releaseDate, m.runtime, m.language,
               rat.rating, rat.timestamp
        FROM movies m
        JOIN r.ratings rat ON m.movieId = rat.movieId
        WHERE rat.userId = ?
        ORDER BY rat.rating DESC
    """, (user_id,))

    movies = []
    for row in cursor.fetchall():
        movie = dict(row)
        movie["genres"] = _parse_genres(movie.get("genres"))
        movies.append(movie)

    movies_conn.close()
    return movies


def get_top_rated_movies(min_ratings: int = 10, limit: int = 50) -> list[dict[str, Any]]:
    """Get top rated movies with at least min_ratings reviews."""
    movies_conn = get_movies_connection()

    # Attach ratings database
    movies_conn.execute(f"ATTACH DATABASE '{RATINGS_DB}' AS r")

    cursor = movies_conn.execute("""
        SELECT m.movieId, m.title, m.overview, m.budget, m.revenue,
               m.genres, m.releaseDate, m.runtime, m.language,
               AVG(rat.rating) as avg_rating,
               COUNT(rat.ratingId) as num_ratings
        FROM movies m
        JOIN r.ratings rat ON m.movieId = rat.movieId
        GROUP BY m.movieId
        HAVING num_ratings >= ?
        ORDER BY avg_rating DESC
        LIMIT ?
    """, (min_ratings, limit))

    movies = []
    for row in cursor.fetchall():
        movie = dict(row)
        movie["genres"] = _parse_genres(movie.get("genres"))
        movies.append(movie)

    movies_conn.close()
    return movies


def search_movies(
    genres: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    min_budget: int | None = None,
    min_revenue: int | None = None,
    limit: int = 20
) -> list[dict[str, Any]]:
    """
    Search movies with various filters.

    Args:
        genres: List of genres to filter by (OR matching)
        min_year: Minimum release year
        max_year: Maximum release year
        min_rating: Minimum average rating
        min_budget: Minimum budget
        min_revenue: Minimum revenue
        limit: Maximum results to return
    """
    movies_conn = get_movies_connection()
    movies_conn.execute(f"ATTACH DATABASE '{RATINGS_DB}' AS r")

    conditions = ["1=1"]
    params: list[Any] = []

    if genres:
        genre_conditions = []
        for genre in genres:
            genre_conditions.append("m.genres LIKE ?")
            params.append(f"%{genre}%")
        conditions.append(f"({' OR '.join(genre_conditions)})")

    if min_year:
        conditions.append("CAST(SUBSTR(m.releaseDate, 1, 4) AS INTEGER) >= ?")
        params.append(min_year)

    if max_year:
        conditions.append("CAST(SUBSTR(m.releaseDate, 1, 4) AS INTEGER) <= ?")
        params.append(max_year)

    if min_budget:
        conditions.append("m.budget >= ?")
        params.append(min_budget)

    if min_revenue:
        conditions.append("m.revenue >= ?")
        params.append(min_revenue)

    # Build query with optional rating filter
    if min_rating:
        query = f"""
            SELECT m.movieId, m.title, m.overview, m.budget, m.revenue,
                   m.genres, m.releaseDate, m.runtime, m.language,
                   AVG(rat.rating) as avg_rating,
                   COUNT(rat.ratingId) as num_ratings
            FROM movies m
            JOIN r.ratings rat ON m.movieId = rat.movieId
            WHERE {' AND '.join(conditions)}
            GROUP BY m.movieId
            HAVING avg_rating >= ?
            ORDER BY avg_rating DESC, num_ratings DESC
            LIMIT ?
        """
        params.extend([min_rating, limit])
    else:
        query = f"""
            SELECT m.movieId, m.title, m.overview, m.budget, m.revenue,
                   m.genres, m.releaseDate, m.runtime, m.language
            FROM movies m
            WHERE {' AND '.join(conditions)}
            ORDER BY m.revenue DESC
            LIMIT ?
        """
        params.append(limit)

    cursor = movies_conn.execute(query, params)

    movies = []
    for row in cursor.fetchall():
        movie = dict(row)
        movie["genres"] = _parse_genres(movie.get("genres"))
        movies.append(movie)

    movies_conn.close()
    return movies


def save_enriched_movie(
    movie_id: int,
    sentiment_score: float,
    budget_tier: str,
    revenue_tier: str,
    roi_category: str,
    themes: list[str],
    model_used: str
) -> None:
    """Save enriched movie data to the database."""
    from datetime import datetime

    conn = get_enriched_connection()
    conn.execute("""
        INSERT OR REPLACE INTO enriched_movies
        (movieId, sentiment_score, budget_tier, revenue_tier,
         roi_category, themes, enriched_at, model_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        movie_id,
        sentiment_score,
        budget_tier,
        revenue_tier,
        roi_category,
        json.dumps(themes),
        datetime.utcnow().isoformat(),
        model_used
    ))
    conn.commit()
    conn.close()


def get_enriched_movie(movie_id: int) -> dict[str, Any] | None:
    """Get enriched data for a movie."""
    conn = get_enriched_connection()
    cursor = conn.execute("""
        SELECT * FROM enriched_movies WHERE movieId = ?
    """, (movie_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    enriched = dict(row)
    if enriched.get("themes"):
        try:
            enriched["themes"] = json.loads(enriched["themes"])
        except json.JSONDecodeError:
            enriched["themes"] = []

    return enriched


def get_all_enriched_movies() -> list[dict[str, Any]]:
    """Get all enriched movies."""
    conn = get_enriched_connection()
    cursor = conn.execute("SELECT * FROM enriched_movies")

    movies = []
    for row in cursor.fetchall():
        movie = dict(row)
        if movie.get("themes"):
            try:
                movie["themes"] = json.loads(movie["themes"])
            except json.JSONDecodeError:
                movie["themes"] = []
        movies.append(movie)

    conn.close()
    return movies


def get_enriched_movies_with_details() -> list[dict[str, Any]]:
    """Get enriched movies joined with movie details."""
    enriched_conn = get_enriched_connection()
    enriched_conn.execute(f"ATTACH DATABASE '{MOVIES_DB}' AS m")

    cursor = enriched_conn.execute("""
        SELECT e.*,
               mov.title, mov.overview, mov.budget, mov.revenue,
               mov.genres, mov.releaseDate, mov.runtime, mov.language
        FROM enriched_movies e
        JOIN m.movies mov ON e.movieId = mov.movieId
    """)

    movies = []
    for row in cursor.fetchall():
        movie = dict(row)
        # Parse themes (simple JSON array)
        if movie.get("themes"):
            try:
                movie["themes"] = json.loads(movie["themes"])
            except json.JSONDecodeError:
                movie["themes"] = []
        # Parse genres (dict format)
        movie["genres"] = _parse_genres(movie.get("genres"))
        movies.append(movie)

    enriched_conn.close()
    return movies
