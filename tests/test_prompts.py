from src.rag.prompts import is_greeting


def test_greeting_chinese():
    assert is_greeting("你好")
    assert is_greeting("早上好！")


def test_greeting_english():
    assert is_greeting("hello")
    assert is_greeting("Hi!")


def test_not_greeting():
    assert not is_greeting("手机保修多久？")
    assert not is_greeting("我要退货")


def test_greeting_plus_question_not_pure_greeting():
    assert not is_greeting("你好，质保政策是什么？")
