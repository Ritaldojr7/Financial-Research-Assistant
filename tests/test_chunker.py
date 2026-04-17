"""Unit tests for the section-aware document chunker.

All tests use pure functions — no external services or mocking needed.
"""
from app.ingestion.chunker import (
    DocumentChunk,
    _split_into_sections,
    _split_paragraphs,
    chunk_text,
)


class TestSectionSplitting:
    def test_detects_common_sections(self):
        doc = (
            "Preamble text here.\n\n"
            "Risk Factors\n\nSome risks.\n\n"
            "Revenue\n\nTotal revenue was $100M.\n\n"
            "Management's Discussion and Analysis\n\nDiscussion here."
        )
        sections = _split_into_sections(doc)
        names = [s.name for s in sections]
        assert "Preamble" in names
        assert "Risk Factors" in names
        assert "Revenue" in names
        assert "Management's Discussion and Analysis" in names

    def test_no_sections_returns_single(self):
        doc = "Just plain text with no headings at all."
        sections = _split_into_sections(doc)
        assert len(sections) == 1
        assert sections[0].name == ""

    def test_section_text_does_not_overlap(self):
        doc = "Risk Factors\n\nRisk A.\n\nRevenue\n\nRevenue was $50M."
        sections = _split_into_sections(doc)
        for s in sections:
            if s.name == "Risk Factors":
                assert "Revenue was" not in s.text
            if s.name == "Revenue":
                assert "Risk A" not in s.text


class TestParagraphSplitting:
    def test_splits_on_double_newline(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        paras = _split_paragraphs(text)
        assert len(paras) == 3

    def test_table_rows_kept_together(self):
        text = (
            "Some text.\n\n"
            "iPhone $200.6B 52.3%\n\n"
            "Mac $29.4B 7.7%\n\n"
            "More text."
        )
        paras = _split_paragraphs(text)
        table_paras = [p for p in paras if "$200.6B" in p or "$29.4B" in p]
        assert len(table_paras) == 1, "Table rows should be merged"


class TestChunkText:
    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []

    def test_basic_chunking_produces_chunks(self):
        text = "Word " * 1000
        chunks = chunk_text(text, company="TestCo", year="2024")
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_metadata_propagated(self):
        text = "Revenue\n\nTotal revenue was $100M for the fiscal year."
        chunks = chunk_text(text, company="Acme", year="2023", source="report.pdf")
        for c in chunks:
            assert c.metadata["company"] == "Acme"
            assert c.metadata["year"] == "2023"
            assert c.metadata["source"] == "report.pdf"

    def test_chunk_index_is_sequential(self):
        text = "Word " * 2000
        chunks = chunk_text(text, company="X", year="2024")
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_section_metadata_assigned(self):
        text = "Risk Factors\n\nThe company faces supply chain risks."
        chunks = chunk_text(text, company="X", year="2024")
        assert any(c.metadata.get("section") == "Risk Factors" for c in chunks)
