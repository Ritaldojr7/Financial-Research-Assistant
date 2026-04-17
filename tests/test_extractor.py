"""Unit tests for the extraction quality scoring helper."""
from app.services.extractor import _count_nulls


class TestCountNulls:
    def test_all_populated(self):
        data = {"income_statement": {"revenue": {"value": 100}}}
        nulls, total = _count_nulls(data)
        assert nulls == 0
        assert total == 1

    def test_all_null(self):
        data = {"income_statement": {"revenue": {"value": None}, "eps_basic": None}}
        nulls, total = _count_nulls(data)
        assert nulls == 2
        assert total == 2

    def test_mixed(self):
        data = {
            "revenue": {"value": 100},
            "net_income": {"value": None},
            "eps_basic": 5.0,
            "eps_diluted": None,
        }
        nulls, total = _count_nulls(data)
        assert total == 4
        assert nulls == 2

    def test_skips_metadata_keys(self):
        data = {
            "company": "Apple",
            "year": "2024",
            "currency": "USD",
            "notes": "some note",
            "revenue": {"value": 100, "unit": "B"},
        }
        nulls, total = _count_nulls(data)
        assert total == 1
        assert nulls == 0

    def test_empty_dict(self):
        nulls, total = _count_nulls({})
        assert nulls == 0
        assert total == 0
