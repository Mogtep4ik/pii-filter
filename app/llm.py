"""LLM-слой (Ollama): два применения маленькой модели.

1. classify_dates — контекстная классификация спорных дат (рождения vs документа),
   когда эвристики не уверены.
2. audit_residual_pii — второй проход: модель проверяет уже очищенный текст и
   возвращает пропущенные PII (типично: названия компаний без ООО/кавычек).

Экономика: модель видит только спорные случаи и короткие фрагменты, а не каждый
документ целиком — основную массу закрывают regex+NER за миллисекунды.
MOCK_MODE=true: слой отключается (для тестов без Ollama).
"""
import os, json, re, httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


async def _generate(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(f"{OLLAMA_URL}/api/generate",
                         json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
                               "options": {"temperature": 0}})
        r.raise_for_status()
        data = r.json()
    usage["llm_calls"] += 1
    usage["prompt_tokens"] += data.get("prompt_eval_count", 0)
    usage["completion_tokens"] += data.get("eval_count", 0)
    return data["response"]


async def classify_dates(text: str, entities) -> list[str]:
    """Для каждой спорной даты вернуть 'birth' или 'keep'."""
    if MOCK_MODE or not entities:
        return ["birth"] * len(entities)  # безопасный дефолт: сомнение -> вычищаем
    items = []
    for i, e in enumerate(entities, 1):
        ctx = text[max(0, e.start - 80):min(len(text), e.end + 40)]
        items.append(f"{i}. дата «{e.value}» в контексте: ...{ctx}...")
    prompt = (
        "Определи для каждой даты: это дата рождения человека (персональные данные) "
        "или дата документа/срок (не персональные данные)?\n"
        + "\n".join(items) +
        "\nОтветь строго JSON-списком из слов birth или keep, например [\"keep\", \"birth\"]."
    )
    out = await _generate(prompt)
    m = re.search(r"\[.*?\]", out, re.S)
    try:
        parsed = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        parsed = []
    verdicts = [v if v in ("birth", "keep") else "birth" for v in parsed]
    verdicts += ["birth"] * (len(entities) - len(verdicts))
    return verdicts[:len(entities)]


async def audit_residual_pii(cleaned_text: str) -> list[dict]:
    """Второй проход: найти PII, пропущенные первым слоем. Возвращает [{type, value}]."""
    if MOCK_MODE:
        return []
    prompt = (
        "Текст уже очищен от персональных данных (плейсхолдеры в [СКОБКАХ] уже заменены — их не трогай). "
        "Найди ОСТАВШИЕСЯ персональные данные: имена/фамилии людей, названия коммерческих компаний и их филиалов, "
        "телефоны, email, адреса проживания, номера документов человека.\n"
        "НЕ считай персональными данными: даты документов и сроки, суммы денег, номера договоров и приложений, "
        "названия законов и госорганов, города в датах и реквизитах документа.\n"
        f"Текст:\n{cleaned_text}\n"
        "Ответь строго JSON-списком объектов вида {\"type\": \"org|fio|phone|email|loc\", \"value\": \"точная строка из текста\"}. "
        "Если ничего не осталось — верни []."
    )
    out = await _generate(prompt)
    m = re.search(r"\[.*\]", out, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    valid = []
    for it in items:
        if isinstance(it, dict) and it.get("value") and it["value"] in cleaned_text:
            valid.append({"type": it.get("type", "org"), "value": it["value"]})
    return valid
