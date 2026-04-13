"""
Pydantic models for structured data and LLM outputs.
"""

from typing import Literal

from pydantic import BaseModel, Field


# Budget tiers based on production budget
BudgetTier = Literal["micro", "low", "medium", "high", "blockbuster"]

# Revenue tiers based on box office performance
RevenueTier = Literal["flop", "underperformer", "moderate", "hit", "blockbuster"]

# ROI categories based on revenue/budget ratio
ROICategory = Literal["disaster", "loss", "break_even", "profitable", "smash_hit"]


class Movie(BaseModel):
    """Base movie model from the database."""

    movieId: int
    title: str
    overview: str | None = None
    budget: int = 0
    revenue: int = 0
    genres: list[str] = Field(default_factory=list)
    releaseDate: str | None = None
    runtime: float | None = None
    language: str | None = None
    imdbId: str | None = None
    status: str | None = None
    productionCompanies: list[str] = Field(default_factory=list)


class EnrichedAttributes(BaseModel):
    """LLM-generated enrichment attributes for a movie."""

    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score of the movie overview. -1.0 (very negative) to 1.0 (very positive)"
    )
    budget_tier: BudgetTier = Field(
        ...,
        description="Budget category: micro (<$1M), low ($1-15M), medium ($15-50M), high ($50-150M), blockbuster (>$150M)"
    )
    revenue_tier: RevenueTier = Field(
        ...,
        description="Revenue category: flop (<$10M), underperformer ($10-50M), moderate ($50-200M), hit ($200-500M), blockbuster (>$500M)"
    )
    roi_category: ROICategory = Field(
        ...,
        description="Return on investment: disaster (<0.5x), loss (0.5-1x), break_even (1-1.5x), profitable (1.5-3x), smash_hit (>3x)"
    )
    themes: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 thematic keywords extracted from the movie overview"
    )


class EnrichedMovie(BaseModel):
    """Movie with LLM-generated enrichment attributes."""

    movieId: int
    sentiment_score: float
    budget_tier: BudgetTier
    revenue_tier: RevenueTier
    roi_category: ROICategory
    themes: list[str]
    enriched_at: str | None = None
    model_used: str | None = None


class MovieWithRating(Movie):
    """Movie with user rating information."""

    rating: float | None = None
    timestamp: int | None = None
    avg_rating: float | None = None
    num_ratings: int | None = None


class MovieRecommendation(BaseModel):
    """A recommended movie with explanation."""

    movie: Movie
    explanation: str = Field(
        ...,
        description="Why this movie is recommended based on the query or user preferences"
    )
    match_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well this movie matches the request (0-1)"
    )


class RecommendationRequest(BaseModel):
    """Request for movie recommendations."""

    query: str | None = Field(
        None,
        description="Natural language query for recommendations"
    )
    user_id: int | None = Field(
        None,
        description="User ID for personalized recommendations"
    )
    compare_movies: list[str] | None = Field(
        None,
        description="Movie titles to compare"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of recommendations to return"
    )


class SearchFilters(BaseModel):
    """Parsed search filters from natural language query."""

    genres: list[str] | None = Field(
        None,
        description="Genres to filter by"
    )
    min_year: int | None = Field(
        None,
        description="Minimum release year"
    )
    max_year: int | None = Field(
        None,
        description="Maximum release year"
    )
    min_rating: float | None = Field(
        None,
        ge=0.5,
        le=5.0,
        description="Minimum average rating"
    )
    sentiment: str | None = Field(
        None,
        description="Desired sentiment: positive, negative, neutral"
    )
    themes: list[str] | None = Field(
        None,
        description="Themes to look for"
    )
    budget_tier: BudgetTier | None = None
    revenue_tier: RevenueTier | None = None


class UserPreferenceSummary(BaseModel):
    """Summary of a user's movie preferences."""

    user_id: int
    favorite_genres: list[str] = Field(
        ...,
        description="User's most-watched genres"
    )
    preferred_themes: list[str] = Field(
        ...,
        description="Common themes in highly-rated movies"
    )
    rating_tendency: str = Field(
        ...,
        description="Whether user tends to rate high, low, or average"
    )
    summary: str = Field(
        ...,
        description="Natural language summary of user's taste"
    )


class MovieComparison(BaseModel):
    """Comparison between two movies."""

    movie1_title: str
    movie2_title: str
    similarities: list[str] = Field(
        ...,
        description="Things the movies have in common"
    )
    differences: list[str] = Field(
        ...,
        description="Key differences between the movies"
    )
    recommendation: str = Field(
        ...,
        description="Which movie might be better for different types of viewers"
    )
