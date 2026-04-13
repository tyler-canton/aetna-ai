"""Tests for database layer."""

import pytest

from src.db import (
    get_movies_connection,
    get_ratings_connection,
    get_movies_for_enrichment,
    get_movie_by_id,
    get_user_ratings,
    get_user_rated_movies,
    get_top_rated_movies,
    search_movies,
    search_movies_by_title,
)


class TestDatabaseConnections:
    """Test database connection functions."""

    def test_movies_connection(self):
        """Test movies database connection."""
        conn = get_movies_connection()
        assert conn is not None

        cursor = conn.execute("SELECT COUNT(*) FROM movies")
        count = cursor.fetchone()[0]
        assert count > 0

        conn.close()

    def test_ratings_connection(self):
        """Test ratings database connection."""
        conn = get_ratings_connection()
        assert conn is not None

        cursor = conn.execute("SELECT COUNT(*) FROM ratings")
        count = cursor.fetchone()[0]
        assert count > 0

        conn.close()


class TestMovieQueries:
    """Test movie query functions."""

    def test_get_movies_for_enrichment(self):
        """Test getting movies suitable for enrichment."""
        movies = get_movies_for_enrichment(limit=10)

        assert len(movies) <= 10
        assert len(movies) > 0

        # Check required fields
        for movie in movies:
            assert "movieId" in movie
            assert "title" in movie
            assert "overview" in movie
            assert "budget" in movie
            assert "revenue" in movie

            # Verify enrichment criteria
            assert movie["overview"] is not None
            assert len(movie["overview"]) > 100
            assert movie["budget"] > 0
            assert movie["revenue"] > 0

    def test_get_movie_by_id(self):
        """Test getting a specific movie."""
        # Get a known movie first
        movies = get_movies_for_enrichment(limit=1)
        if movies:
            movie_id = movies[0]["movieId"]
            movie = get_movie_by_id(movie_id)

            assert movie is not None
            assert movie["movieId"] == movie_id
            assert "title" in movie
            assert "overview" in movie

    def test_get_movie_by_id_not_found(self):
        """Test getting a non-existent movie."""
        movie = get_movie_by_id(999999999)
        assert movie is None

    def test_search_movies_by_title(self):
        """Test searching movies by title."""
        movies = search_movies_by_title("Avatar", limit=5)

        assert len(movies) > 0
        # At least one should contain "Avatar"
        titles = [m["title"].lower() for m in movies]
        assert any("avatar" in t for t in titles)

    def test_search_movies_with_filters(self):
        """Test searching movies with various filters."""
        # Search by genre
        movies = search_movies(genres=["Action"], limit=10)
        assert len(movies) > 0

        # Search by year range
        movies = search_movies(min_year=2010, max_year=2015, limit=10)
        assert len(movies) > 0

        # Search by minimum rating
        movies = search_movies(min_rating=4.0, limit=10)
        assert len(movies) > 0

    def test_get_top_rated_movies(self):
        """Test getting top rated movies."""
        movies = get_top_rated_movies(min_ratings=10, limit=10)

        assert len(movies) > 0
        for movie in movies:
            assert "avg_rating" in movie
            assert "num_ratings" in movie
            assert movie["num_ratings"] >= 10


class TestRatingQueries:
    """Test rating query functions."""

    def test_get_user_ratings(self):
        """Test getting user ratings."""
        # User 1 should have ratings
        ratings = get_user_ratings(user_id=1)

        assert len(ratings) > 0
        for rating in ratings:
            assert rating["userId"] == 1
            assert "movieId" in rating
            assert "rating" in rating
            assert 0.5 <= rating["rating"] <= 5.0

    def test_get_user_rated_movies(self):
        """Test getting movies rated by a user."""
        movies = get_user_rated_movies(user_id=1)

        assert len(movies) > 0
        for movie in movies:
            assert "movieId" in movie
            assert "title" in movie
            assert "rating" in movie

    def test_get_user_ratings_no_ratings(self):
        """Test getting ratings for user with no ratings."""
        ratings = get_user_ratings(user_id=999999)
        assert len(ratings) == 0
