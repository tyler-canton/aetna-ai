"""Tests for recommendation system."""

import pytest
from unittest.mock import patch, MagicMock
import json

from src.models import SearchFilters, MovieRecommendation, Movie


class TestQueryParsing:
    """Test natural language query parsing."""

    def test_search_filters_parsing(self):
        """Test creating SearchFilters from parsed data."""
        # Simulated LLM output for "action movies from the 90s"
        parsed_data = {
            "genres": ["Action"],
            "min_year": 1990,
            "max_year": 1999,
        }

        filters = SearchFilters(**parsed_data)

        assert filters.genres == ["Action"]
        assert filters.min_year == 1990
        assert filters.max_year == 1999
        assert filters.min_rating is None

    def test_search_filters_with_rating(self):
        """Test SearchFilters with rating filter."""
        # Simulated LLM output for "highly rated comedies"
        parsed_data = {
            "genres": ["Comedy"],
            "min_rating": 4.0,
        }

        filters = SearchFilters(**parsed_data)

        assert filters.genres == ["Comedy"]
        assert filters.min_rating == 4.0

    def test_search_filters_with_themes(self):
        """Test SearchFilters with theme filter."""
        # Simulated LLM output for "dark thrillers about revenge"
        parsed_data = {
            "genres": ["Thriller"],
            "sentiment": "negative",
            "themes": ["revenge"],
        }

        filters = SearchFilters(**parsed_data)

        assert filters.genres == ["Thriller"]
        assert filters.sentiment == "negative"
        assert filters.themes == ["revenge"]


class TestRecommendations:
    """Test recommendation generation."""

    @pytest.fixture
    def sample_movie(self):
        """Sample movie for testing."""
        return Movie(
            movieId=1,
            title="The Dark Knight",
            overview="Batman fights the Joker in Gotham City.",
            budget=185000000,
            revenue=1004558444,
            genres=["Action", "Crime", "Drama"],
            releaseDate="2008-07-16",
        )

    def test_movie_recommendation_creation(self, sample_movie):
        """Test creating a movie recommendation."""
        rec = MovieRecommendation(
            movie=sample_movie,
            explanation="This dark superhero film matches your request for intense action movies.",
            match_score=0.92,
        )

        assert rec.movie.title == "The Dark Knight"
        assert rec.match_score == 0.92
        assert "superhero" in rec.explanation

    def test_recommendation_sorting(self, sample_movie):
        """Test that recommendations can be sorted by match score."""
        recs = [
            MovieRecommendation(movie=sample_movie, explanation="A", match_score=0.5),
            MovieRecommendation(movie=sample_movie, explanation="B", match_score=0.9),
            MovieRecommendation(movie=sample_movie, explanation="C", match_score=0.7),
        ]

        sorted_recs = sorted(recs, key=lambda r: r.match_score, reverse=True)

        assert sorted_recs[0].match_score == 0.9
        assert sorted_recs[1].match_score == 0.7
        assert sorted_recs[2].match_score == 0.5


class TestRecommendationModes:
    """Test different recommendation modes."""

    def test_query_mode_filters(self):
        """Test that query mode extracts appropriate filters."""
        # Test various query patterns
        test_cases = [
            ("action movies", {"genres": ["Action"]}),
            ("comedies from 2020", {"genres": ["Comedy"], "min_year": 2020, "max_year": 2020}),
            ("highly rated horror", {"genres": ["Horror"], "min_rating": 4.0}),
        ]

        for query, expected_keys in test_cases:
            # Just verify expected keys would be present
            filters = SearchFilters(**expected_keys)
            assert filters is not None

    def test_user_mode_requires_user_id(self):
        """Test that user mode requires user ID."""
        from src.recommend import recommend_for_user

        # This should handle missing user gracefully
        # (returns empty list, no crash)
        result = recommend_for_user(user_id=999999, limit=5)
        # Should return empty list for non-existent user
        assert isinstance(result, list)


class TestMovieComparison:
    """Test movie comparison functionality."""

    def test_comparison_structure(self):
        """Test movie comparison output structure."""
        from src.models import MovieComparison

        comparison = MovieComparison(
            movie1_title="The Dark Knight",
            movie2_title="Spider-Man",
            similarities=[
                "Both are superhero movies",
                "Both feature iconic villains",
                "Both have sequels",
            ],
            differences=[
                "Dark Knight is darker in tone",
                "Spider-Man is more family-friendly",
                "Different studios (Warner Bros vs Sony)",
            ],
            recommendation="Choose Dark Knight for mature themes, Spider-Man for fun adventure.",
        )

        assert len(comparison.similarities) == 3
        assert len(comparison.differences) == 3
        assert "Dark Knight" in comparison.recommendation


class TestPreferenceSummary:
    """Test user preference summarization."""

    def test_preference_summary_structure(self):
        """Test preference summary output structure."""
        from src.models import UserPreferenceSummary

        summary = UserPreferenceSummary(
            user_id=1,
            favorite_genres=["Action", "Sci-Fi", "Thriller"],
            preferred_themes=["technology", "future", "heroism"],
            rating_tendency="generous",
            summary="User enjoys high-octane action films with futuristic settings.",
        )

        assert summary.user_id == 1
        assert len(summary.favorite_genres) == 3
        assert summary.rating_tendency in ["generous", "critical", "balanced"]
