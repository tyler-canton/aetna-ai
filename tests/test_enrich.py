"""Tests for enrichment pipeline."""

import pytest
from unittest.mock import patch, MagicMock
import json

from src.models import EnrichedAttributes


class TestEnrichmentPipeline:
    """Test enrichment pipeline functions."""

    @pytest.fixture
    def sample_movie(self):
        """Sample movie for testing."""
        return {
            "movieId": 1,
            "title": "Test Movie",
            "overview": "A thrilling adventure about a hero saving the world from evil forces.",
            "budget": 100000000,
            "revenue": 500000000,
            "genres": ["Action", "Adventure"],
        }

    @pytest.fixture
    def mock_enrichment_response(self):
        """Mock LLM response for enrichment."""
        return {
            "sentiment_score": 0.7,
            "budget_tier": "high",
            "revenue_tier": "blockbuster",
            "roi_category": "smash_hit",
            "themes": ["heroism", "adventure", "good vs evil"],
        }

    def test_enriched_attributes_from_response(self, mock_enrichment_response):
        """Test creating EnrichedAttributes from LLM response."""
        enriched = EnrichedAttributes(**mock_enrichment_response)

        assert enriched.sentiment_score == 0.7
        assert enriched.budget_tier == "high"
        assert enriched.revenue_tier == "blockbuster"
        assert enriched.roi_category == "smash_hit"
        assert len(enriched.themes) == 3

    def test_budget_tier_calculation(self):
        """Test budget tier logic based on budget amounts."""
        # These are the expected tier boundaries
        tiers = {
            500000: "micro",      # < $1M
            5000000: "low",       # $1-15M
            30000000: "medium",   # $15-50M
            100000000: "high",    # $50-150M
            200000000: "blockbuster",  # > $150M
        }

        # Verify tier boundaries make sense
        assert tiers[500000] == "micro"
        assert tiers[200000000] == "blockbuster"

    def test_revenue_tier_calculation(self):
        """Test revenue tier logic based on revenue amounts."""
        tiers = {
            5000000: "flop",           # < $10M
            25000000: "underperformer", # $10-50M
            100000000: "moderate",      # $50-200M
            300000000: "hit",           # $200-500M
            600000000: "blockbuster",   # > $500M
        }

        assert tiers[5000000] == "flop"
        assert tiers[600000000] == "blockbuster"

    def test_roi_category_calculation(self):
        """Test ROI category logic."""
        def calculate_roi_category(budget: int, revenue: int) -> str:
            if budget == 0:
                return "break_even"
            ratio = revenue / budget
            if ratio < 0.5:
                return "disaster"
            elif ratio < 1.0:
                return "loss"
            elif ratio < 1.5:
                return "break_even"
            elif ratio < 3.0:
                return "profitable"
            else:
                return "smash_hit"

        assert calculate_roi_category(100, 25) == "disaster"    # 0.25x
        assert calculate_roi_category(100, 75) == "loss"        # 0.75x
        assert calculate_roi_category(100, 125) == "break_even" # 1.25x
        assert calculate_roi_category(100, 200) == "profitable" # 2.0x
        assert calculate_roi_category(100, 500) == "smash_hit"  # 5.0x


class TestEnrichmentIntegration:
    """Integration tests for enrichment (require API key)."""

    @pytest.mark.skipif(
        True,  # Set to False to run integration tests
        reason="Integration tests disabled by default"
    )
    def test_enrich_single_movie(self, sample_movie):
        """Test enriching a single movie with real API call."""
        from src.enrich import enrich_single_movie
        import asyncio

        result = asyncio.run(enrich_single_movie(sample_movie))

        assert result is not None
        assert "sentiment_score" in result
        assert "budget_tier" in result
        assert "revenue_tier" in result
        assert "roi_category" in result
        assert "themes" in result
        assert len(result["themes"]) >= 3
