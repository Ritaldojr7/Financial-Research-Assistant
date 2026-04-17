"""Unit tests for comparison detection and company extraction.

Pure function tests — no external services or mocking needed.
"""
from app.services.analyzer import extract_companies, is_comparison_query


class TestComparisonDetection:
    def test_compare_keyword(self):
        assert is_comparison_query("Compare Apple and Microsoft revenue")

    def test_vs_keyword(self):
        assert is_comparison_query("Apple vs Microsoft")

    def test_versus_keyword(self):
        assert is_comparison_query("Apple versus Microsoft earnings")

    def test_difference_between(self):
        assert is_comparison_query("What is the difference between Apple and Google?")

    def test_plain_query_not_detected(self):
        assert not is_comparison_query("What was Apple revenue in 2024?")

    def test_case_insensitive(self):
        assert is_comparison_query("COMPARE apple AND microsoft")


class TestCompanyExtraction:
    def test_extracts_two_companies(self):
        companies = extract_companies("Compare Apple and Microsoft")
        assert len(companies) == 2
        assert "Apple" in companies
        assert "Microsoft" in companies

    def test_extracts_with_vs(self):
        companies = extract_companies("Compare Tesla and Ford")
        assert "Tesla" in companies
        assert "Ford" in companies

    def test_no_match_returns_empty(self):
        companies = extract_companies("What was Apple revenue?")
        assert companies == []
