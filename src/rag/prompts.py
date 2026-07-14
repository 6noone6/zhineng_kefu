import re

LANGUAGE_MATCH_INSTRUCTION = (
    "你是跨境电商智能客服。"
    "始终使用与用户相同的语言回复（用户用英文则英文，用中文则中文，用阿拉伯语则阿拉伯语，以此类推）。"
    "You are a cross-border e-commerce assistant. "
    "Always respond in the same language as the user's message."
)

_PINYIN_GREETING_HINT = (
    "常见拼音问候等同于中文：nihao→你好，xiexie→谢谢，zaoshanghao→早上好。"
)

AGENT_SYSTEM_PROMPT = (
    f"{LANGUAGE_MATCH_INSTRUCTION}\n"
    f"{_PINYIN_GREETING_HINT}\n"
    "若用户仅为问候（中文、英文或拼音形式），友好回复即可，不要查询知识库。"
)

TOOL_SELECTION_FEW_SHOT = """
边界示例（仅供判断，不要复述示例原文）：
- 用户: "你好" → 不调用工具（纯问候）
- 用户: "你好，手机保修多久？" → customer_chat（问候+咨询，以咨询为主）
- 用户: "我要退货" → create_return_request（退换货诉求，不是投诉）
- 用户: "客服态度太差了" → record_user_complaint（服务质量投诉）
- 用户: "我的订单到哪了？"（已登录）→ query_my_orders；若返回 tracking_number 且问物流，可同时 fetch_logistics_information
- 用户: "查订单 ORD-1001 物流" → 可同时 query_order 与 fetch_logistics_information（若已知单号则直接查物流）
"""

TOOL_SELECTION_HINT = (
    f"{_PINYIN_GREETING_HINT}\n"
    "若用户消息仅为问候（如 你好、hello、hi、nihao），不要调用任何工具，"
    "直接输出 {\"tool\": null}。"
    "不要将问候语当作产品或政策咨询传给 customer_chat。"
    "若用户要退货、退款、换货或申请退换货（如 帮我退一下货、我要退款），"
    "使用 create_return_request，不要使用 record_user_complaint。"
    "record_user_complaint 仅用于服务质量投诉（如态度差、未收到货），"
    "不用于退换货政策咨询或退货申请。"
    "若用户问「我的订单」「my orders」且已登录，使用 query_my_orders。"
    "复杂问题可连续调用多个工具；若需订单与物流，可在同一轮并行调用 query_order 与 fetch_logistics_information。"
    f"{TOOL_SELECTION_FEW_SHOT}"
)

REACT_SYSTEM_HINT = (
    "You are a cross-border customer service agent with tools. "
    "Use tools step by step until you have enough information to answer. "
    "After query_order or query_my_orders returns a tracking_number, "
    "call fetch_logistics_information with that number if user asks about shipping. "
    "When no more tools are needed, stop calling tools."
)

_GREETING_RE = re.compile(
    r"^(?:"
    r"你好|您好|哈喽|嗨|嗨喽|"
    r"hello|hi|hey|howdy|"
    r"good\s+(?:morning|afternoon|evening)|"
    r"nihao|ni\s*hao|nǐ\s*hǎo|"
    r"你好啊|早上好|下午好|晚上好|"
    r"thanks|thank\s+you|xiexie|xie\s*xie|"
    r"مرحبا|السلام\s*عليكم|أهلا"
    r")[\s!.?，,~！]*$",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.match(text.strip()))


def build_greeting_messages(text: str) -> list[dict]:
    lang = detect_user_language(text)
    if lang == "en":
        system = (
            "You are a cross-border e-commerce customer service assistant. "
            "The user sent a short greeting. Reply warmly in English only. "
            "Do not use Chinese. Do not query or cite the knowledge base."
        )
    elif lang == "ar":
        system = (
            "أنت مساعد خدمة عملاء للتجارة الإلكترونية عبر الحدود. "
            "رد بتحية دافئة بالعربية فقط. لا تستخدم لغات أخرى."
        )
    else:
        system = AGENT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text.strip()},
    ]


def detect_user_language(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[a-zA-Z]", text):
        return "en"
    return "zh"


def language_reply_instruction(lang: str) -> str:
    if lang == "en":
        return (
            "The user wrote in English. You MUST reply entirely in English. "
            "Translate or summarize any Chinese knowledge into English."
        )
    if lang == "ar":
        return (
            "The user wrote in Arabic. You MUST reply entirely in Arabic. "
            "Translate or summarize any knowledge into Arabic."
        )
    return LANGUAGE_MATCH_INSTRUCTION

RAG_SYSTEM_PROMPT = """你是一个跨境智能客服，请根据以下知识回答用户问题。

知识内容：
{informations}

要求：
1. 只能依据上述知识回答
2. 找不到对应内容时，用与用户相同的语言告知无法找到
3. 不要回答参考资料之外的内容
4. 知识库含中文、英文、阿拉伯语文档；优先采用与用户语言一致的片段，其余内容翻译后再作答
5. 最终回复只能使用一种语言（与用户相同），不得在回复中夹杂其他语言
6. 只输出对用户当前问题的直接回答，不要输出对话历史、多轮格式或 user/assistant 角色标记
7. {language_instruction}
"""


def build_rag_messages(query: str, informations: list[str]) -> list[dict]:
    info_text = "\n---\n".join(informations)
    lang = detect_user_language(query)
    lang_instruction = language_reply_instruction(lang)
    user_content = query
    if lang == "en":
        user_content = f"Answer in English only.\n\n{query}"
    elif lang == "ar":
        user_content = f"Answer in Arabic only.\n\n{query}"
    return [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT.format(
                informations=info_text,
                language_instruction=lang_instruction,
            ),
        },
        {"role": "user", "content": user_content},
    ]
