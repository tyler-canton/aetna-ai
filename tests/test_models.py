"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from src.models import (
    Movie,
    EnrichedAttributes,
    EnrichedMovie,
    MovieWithRating,
    MovieRecommendation,
    RecommendationRequest,
    SearchFilters,
    UserPreferenceSummary,
    MovieComparison,
)


class TestMovieModel:
    """Test Movie model."""

    def test_movie_minimal(self):
        """Test movie with minimal fields."""
        movie = Movie(movieId=1, title="Test Movie")
        assert movie.movieId == 1
        assert movie.title == "Test Movie"
        assert movie.genres == []

    def test_movie_full(self):
        """Test movie with all fields."""
        movie = Movie(
            movieId=1,
            title="Test Movie",
            overview="A great movie",
            budget=1000000,
            revenue=5000000,
            genres=["Action", "Drama"],
            releaseDate="2020-01-01",
            runtime=120.0,
            language="en",
        )
        assert movie.budget == 1000000
        assert movie.revenue == 5000000
        assert len(movie.genres) == 2


class TestEnrichedAttributes:
    """Test EnrichedAttributes model."""

    def test_valid_enrichment(self):
        """Test valid enrichment attributes."""
        enriched = EnrichedAttributes(
            sentiment_score=0.5,
            budget_tier="high",
            revenue_tier="blockbuster",
            roi_category="smash_hit",
            themes=["action", "adventure", "heroism"],
        )
        assert enriched.sentiment_score == 0.5
        assert enriched.budget_tier == "high"
        assert len(enriched.themes) == 3

    def test_sentiment_score_bounds(self):
        """Test sentiment score validation."""
        # Valid bounds
        EnrichedAttributes(
            sentiment_score=-1.0,
            budget_tier="low",
            revenue_tier="flop",
            roi_category="disaster",
            themes=["a", "b", "c"],
        )
        EnrichedAttributes(
            sentiment_score=1.0,
            budget_tier="low",
            revenue_tier="flop",
            roi_category="disaster",
            themes=["a", "b", "c"],
        )

        # Invalid bounds
        with pytest.raises(ValidationError):
            EnrichedAttributes(
                sentiment_score=-1.5,  # Too low
                budget_tier="low",
                revenue_tier="flop",
                roi_category="disaster",
                themes=["a", "b", "c"],
            )

        with pytest.raises(ValidationError):
            EnrichedAttributes(
                sentiment_score=1.5,  # Too high
                budget_tier="low",
                revenue_tier="flop",
                roi_category="disaster",
                themes=["a", "b", "c"],
            )

    def test_invalid_budget_tier(self):
        """Test invalid budget tier."""
        with pytest.raises(ValidationError):
            EnrichedAttributes(
                sentiment_score=0.0,
                budget_tier="invalid",  # Not a valid tier
                revenue_tier="flop",
                roi_category="disaster",
                themes=["a", "b", "c"],
            )

    def test_themes_count(self):
        """Test themes count validation."""
        # Too few themes
        with pytest.raises(ValidationError):
            EnrichedAttributes(
                sentiment_score=0.0,
                budget_tier="low",
                revenue_tier="flop",
                roi_category="disaster",
                themes=["a", "b"],  # Only 2
            )

        # Too many themes
        with pytest.raises(ValidationError):
            EnrichedAttributes(
                sentiment_score=0.0,
                budget_tier="low",
                revenue_tier="flop",
                roi_category="disaster",
                themes=["a", "b", "c", "d", "e", "f"],  # 6 themes
            )


class TestSearchFilters:
    """Test SearchFilters model."""

    def test_empty_filters(self):
        """Test empty filters."""
        filters = SearchFilters()
        assert filters.genres is None
        assert filters.min_year is None
        assert filters.min_rating is None

    def test_partial_filters(self):
        """Test partial filters."""
        filters = SearchFilters(
            genres=["Action", "Comedy"],
            min_year=2000,
        )
        assert filters.genres == ["Action", "Comedy"]
        assert filters.min_year == 2000
        assert filters.max_year is None

    def test_rating_bounds(self):
        """Test rating bounds."""
        filters = SearchFilters(min_rating=4.5)
        assert filters.min_rating == 4.5

        with pytest.raises(ValidationError):
            SearchFilters(min_rating=6.0)  # Above 5.0

        with pytest.raises(ValidationError):
            SearchFilters(min_rating=0.0)  # Below 0.5


class TestRecommendationRequest:
    """Test RecommendationRequest model."""

    def test_query_request(self):
        """Test query-based request."""
        req = RecommendationRequest(query="action movies")
        assert req.query == "action movies"
        assert req.user_id is None
        assert req.limit == 10

    def test_user_request(self):
        """Test user-based request."""
        req = RecommendationRequest(user_id=42, limit=5)
        assert req.user_id == 42
        assert req.limit == 5

    def test_compare_request(self):
        """Test comparison request."""
        req = RecommendationRequest(compare_movies=["Movie A", "Movie B"])
        assert len(req.compare_movies) == 2

    def test_limit_bounds(self):
        """Test limit bounds."""
        with pytest.raises(ValidationError):
            RecommendationRequest(query="test", limit=0)

        with pytest.raises(ValidationError):
            RecommendationRequest(query="test", limit=100)


class TestMovieRecommendation:
    """Test MovieRecommendation model."""

    def test_valid_recommendation(self):
        """Test valid recommendation."""
        movie = Movie(movieId=1, title="Test")
        rec = MovieRecommendation(
            movie=movie,
            explanation="Great match for your preferences",
            match_score=0.85,
        )
        assert rec.match_score == 0.85
        assert "preferences" in rec.explanation

    def test_match_score_bounds(self):
        """Test match score bounds."""
        movie = Movie(movieId=1, title="Test")

        with pytest.raises(ValidationError):
            MovieRecommendation(
                movie=movie,
                explanation="Test",
                match_score=1.5,  # Above 1.0
            )

        with pytest.raises(ValidationError):
            MovieRecommendation(
                movie=movie,
                explanation="Test",
                match_score=-0.5,  # Below 0.0
            )


class TestUserPreferenceSummary:
    """Test UserPreferenceSummary model."""

    def test_valid_summary(self):
        """Test valid preference summary."""
        summary = UserPreferenceSummary(
            user_id=1,
            favorite_genres=["Action", "Comedy", "Drama"],
            preferred_themes=["adventure", "humor", "friendship"],
            rating_tendency="generous",
            summary="User enjoys action and comedy films with positive themes.",
        )
        assert summary.user_id == 1
        assert len(summary.favorite_genres) == 3
        assert summary.rating_tendency == "generous"


class TestMovieComparison:
    """Test MovieComparison model."""

    def test_valid_comparison(self):
        """Test valid movie comparison."""
        comparison = MovieComparison(
            movie1_title="The Dark Knight",
            movie2_title="Spider-Man",
            similarities=["superhero genre", "action sequences", "villain focus"],
            differences=["tone (dark vs light)", "hero type", "studio"],
            recommendation="Dark Knight for mature viewers, Spider-Man for families",
        )
        assert comparison.movie1_title == "The Dark Knight"
        assert len(comparison.similarities) == 3
        assert len(comparison.differences) == 3
