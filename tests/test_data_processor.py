#!/usr/bin/env python3
"""
tests/test_data_processor.py – Unit tests for the data processing pipeline.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processor import (
    get_simplified_label,
    cord_sample_to_ner,
    regex_extract,
    LABEL_LIST,
    LABEL2ID,
)


class TestGetSimplifiedLabel:
    def test_total_label(self):
        assert get_simplified_label("total.total_price") == "TOTAL"

    def test_date_label(self):
        assert get_simplified_label("sub_total.datetime") == "DATE"

    def test_vendor_label(self):
        assert get_simplified_label("sub_total.etc") == "VENDOR"

    def test_unknown_label(self):
        assert get_simplified_label("some.unknown.field") is None

    def test_menu_item_not_key_field(self):
        # ITEM is not a key field we care about
        result = get_simplified_label("menu.nm")
        assert result not in ("TOTAL", "DATE", "VENDOR", "ID")


class TestLabelList:
    def test_all_bio_tags_present(self):
        for tag in ("O", "B-TOTAL", "I-TOTAL", "B-DATE", "I-DATE"):
            assert tag in LABEL_LIST

    def test_label2id_consistent(self):
        for label, idx in LABEL2ID.items():
            assert LABEL_LIST[idx] == label


class TestRegexExtract:
    def test_extract_total(self):
        text = "TOTAL: $25.50\nThank you"
        result = regex_extract(text)
        assert result["total_amount"] is not None
        assert "25" in result["total_amount"]

    def test_extract_date_slash_format(self):
        text = "Date: 15/03/2024\nItems: ..."
        result = regex_extract(text)
        assert result["date"] is not None
        assert "2024" in result["date"]

    def test_extract_date_dash_format(self):
        text = "Invoice Date: 2024-01-15"
        result = regex_extract(text)
        assert result["date"] is not None

    def test_extract_receipt_id(self):
        text = "Receipt No: RC-20240115-001"
        result = regex_extract(text)
        assert result["receipt_id"] is not None

    def test_empty_text(self):
        result = regex_extract("")
        assert all(v is None for v in result.values())

    def test_grand_total_keyword(self):
        text = "Grand Total: 36.30"
        result = regex_extract(text)
        assert result["total_amount"] is not None


class TestCordSampleToNer:
    def test_basic_sample(self):
        sample = {
            "id": 1,
            "ground_truth": json.dumps({
                "valid_line": [
                    {
                        "category": "total.total_price",
                        "words": [
                            {"quad_text": "25.50", "quad": {}}
                        ]
                    },
                    {
                        "category": "menu.nm",
                        "words": [
                            {"quad_text": "Coffee", "quad": {}},
                            {"quad_text": "Muffin", "quad": {}}
                        ]
                    }
                ]
            })
        }
        result = cord_sample_to_ner(sample)
        assert len(result["tokens"]) >= 1
        assert len(result["tokens"]) == len(result["ner_tags"])

    def test_empty_annotation(self):
        sample = {"id": 2, "ground_truth": "{}"}
        result = cord_sample_to_ner(sample)
        assert result["tokens"] == []
        assert result["ner_tags"] == []
