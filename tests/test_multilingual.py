from src.rag.multilingual import (
    build_cross_lingual_queries,
    detect_chunk_language,
)
from src.services.workflows.return_workflow import (
    extract_order_id,
    is_return_intent,
)


def test_detect_chunk_language_en_suffix():
    assert detect_chunk_language("returns_refund_en.txt", "中文") == "en"


def test_detect_chunk_language_ar_suffix():
    assert detect_chunk_language("faq_contact_ar.txt", "text") == "ar"


def test_build_cross_lingual_queries_english():
    queries = build_cross_lingual_queries("How long is warranty?", "en")
    assert len(queries) >= 2
    assert queries[0] == "How long is warranty?"


def test_return_intent_chinese():
    assert is_return_intent("我要退货")


def test_return_intent_english():
    assert is_return_intent("I need a refund")


def test_extract_order_id():
    assert extract_order_id("查订单 ORD-1001 物流") == "ORD-1001"
