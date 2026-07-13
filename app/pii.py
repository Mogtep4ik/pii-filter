"""Пайплайн очистки PII: три слоя.

1. Регексы — однозначные машинные форматы: телефон, email, СНИЛС, ИНН, счёт, карта, паспорт.
2. NER (Natasha) — русские ФИО, организации (+ эвристика на «филиал»).
3. Контекстный слой для дат: дата рождения — PII (вычищаем), дата подписания/срок — нет
   (оставляем). Решается по контексту: сначала быстрые эвристики, спорные случаи — LLM.

Каждая найденная сущность заменяется плейсхолдером вида [ТЕЛЕФОН_1], [ФИО_2] — так текст
остаётся читаемым для внешней LLM, а связи между упоминаниями сохраняются (одинаковое
значение -> одинаковый плейсхолдер).
"""
import re
from dataclasses import dataclass

from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, Doc

# ---------- singleton-модели natasha (грузятся один раз при импорте) ----------
_segmenter = Segmenter()
_morph = MorphVocab()
_emb = NewsEmbedding()
_ner = NewsNERTagger(_emb)

PLACEHOLDER_RU = {
    "phone": "ТЕЛЕФОН", "email": "EMAIL", "snils": "СНИЛС", "inn": "ИНН",
    "account": "СЧЁТ", "card": "КАРТА", "passport": "ПАСПОРТ",
    "fio": "ФИО", "org": "ОРГАНИЗАЦИЯ", "branch": "ФИЛИАЛ", "loc": "АДРЕС",
    "birthdate": "ДАТА_РОЖДЕНИЯ",
}

# ---------- слой 1: регексы ----------
REGEXES = [
    ("email", re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")),
    ("phone", re.compile(r"(?<!\d)(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)")),
    ("snils", re.compile(r"\b\d{3}-\d{3}-\d{3}[\s-]?\d{2}\b")),
    ("account", re.compile(r"\b40[0-9]{18}\b")),          # р/с начинается с 40...
    ("card", re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b")),
    ("passport", re.compile(r"\b\d{2}\s?\d{2}\s\d{6}\b")),
    ("inn", re.compile(r"(?<=ИНН[\s:])\s*\d{10,12}\b|\b\d{12}\b|(?<!\d)\d{10}(?!\d)")),
]

DATE_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{1,2}\s+(?:янв|фев|мар|апр|ма[яй]|июн|июл|авг|сен|окт|ноя|дек)[а-я]*\s+\d{4}(?:\s+года)?", re.I)

BIRTH_CTX = re.compile(r"рожден|род\.|дата\s+рожд|д\.р\.", re.I)
DOC_DATE_CTX = re.compile(r"договор|подписа|заключ|составлен|акт|срок|не\s+позднее|до\s*$|вступает|выдан|запланирован|отпуск|принята.*до|закон[а-я]*\s+от|приказ[а-я]*|исх\.?\s*№|вх\.?\s*№|постановлен|№\s*\d+[-\s]?ФЗ", re.I)

BRANCH_RE = re.compile(r"(?:[А-ЯЁ][а-яё-]+\s+)?филиал[а-яё]*(?:\s+«[^»]+»|\s+в\s+г\.\s*[А-ЯЁ][а-яё-]+)?|обособленное\s+подразделение\s+«[^»]+»", re.I)
ORG_QUOTED_RE = re.compile(r"\b(?:ООО|АО|ПАО|ЗАО|ИП|ФГУП|ГУП|АНО)\s+«[^»]+»")


@dataclass
class Entity:
    kind: str
    start: int
    end: int
    value: str


def _find_regex_entities(text: str) -> list[Entity]:
    found = []
    for kind, rx in REGEXES:
        for m in rx.finditer(text):
            found.append(Entity(kind, m.start(), m.end(), m.group(0).strip()))
    for m in BRANCH_RE.finditer(text):
        found.append(Entity("branch", m.start(), m.end(), m.group(0)))
    for m in ORG_QUOTED_RE.finditer(text):
        found.append(Entity("org", m.start(), m.end(), m.group(0)))
    return found


def _find_ner_entities(text: str) -> list[Entity]:
    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_ner(_ner)
    out = []
    NOT_PII_LOC = {"рф", "россия", "россии", "российская федерация", "российской федерации"}
    # заглавные договорные термины и прочие ложные цели NER в казённых текстах
    NER_STOPWORDS = ("заказчик", "поставщик", "исполнител", "подрядчик", "сторон", "контракт",
                     "договор", "товар", "спецификаци", "извещени", "распоряжени", "приложени",
                     "техническ", "требовани", "ктру", "окпд", "закон", "правительств",
                     "синий", "зелен", "голуб", "коричнев", "красн", "черн", "бел", "масля",
                     "м.п", "при исполнении", "да")
    for span in doc.spans:
        kind = {"PER": "fio", "ORG": "org", "LOC": "loc"}.get(span.type)
        if not kind:
            continue
        val = text[span.start:span.stop]
        if kind == "loc" and (val.lower() in NOT_PII_LOC
                              or re.search(r"(?:ст\.|ГК|УК|НК|ТК|кодекс[а-я]*)\s*$", text[max(0, span.start-12):span.start])):
            continue
        vl = val.lower().strip()
        # мусорные спаны: переносы строк внутри, договорные термины, односложный стоп-лист
        if "\n" in val:
            continue
        words = vl.split()
        if words and all(any(w.startswith(sw) for sw in NER_STOPWORDS) for w in words):
            continue
        out.append(Entity(kind, span.start, span.stop, val))
    return out


def _valid_date(s: str) -> bool:
    """dd.mm.yyyy-подобное: день 1-31, месяц 1-12. Отсекает коды ОКПД2 (17.23.11 и т.п.)."""
    m = re.match(r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})$", s)
    if not m:
        return True  # словесные даты ("15 марта 2024") валидируются самим регексом
    d, mo = int(m.group(1)), int(m.group(2))
    return 1 <= d <= 31 and 1 <= mo <= 12


def _classify_dates_heuristic(text: str) -> list[tuple[Entity, str]]:
    """Возвращает [(entity, 'birth'|'keep'|'unsure')] для каждой даты по контексту вокруг."""
    results = []
    for m in DATE_RE.finditer(text):
        if not _valid_date(m.group(0)):
            continue
        left = text[max(0, m.start() - 60):m.start()]
        right = text[m.end():m.end() + 30]
        ent = Entity("birthdate", m.start(), m.end(), m.group(0))
        # решает БЛИЖАЙШЕЕ к дате ключевое слово, а не любое в окне:
        # «...рождения, паспорт выдан 17.08.2026» -> «выдан» ближе -> keep
        def nearest(rx, seg, from_right=False):
            hits = list(rx.finditer(seg))
            if not hits:
                return None
            h = hits[-1] if not from_right else hits[0]
            return (len(seg) - h.end()) if not from_right else h.start()
        b = min(x for x in [nearest(BIRTH_CTX, left), nearest(BIRTH_CTX, right, True), 10**9] if x is not None)
        k = min(x for x in [nearest(DOC_DATE_CTX, left), 10**9] if x is not None)
        if b == 10**9 and k == 10**9:
            results.append((ent, "unsure"))
        elif b < k:
            results.append((ent, "birth"))
        else:
            results.append((ent, "keep"))
    return results


def _merge(entities: list[Entity]) -> list[Entity]:
    """Убирает перекрытия: приоритет более длинной сущности (org «...» длиннее ФИО внутри)."""
    entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
    merged: list[Entity] = []
    for e in entities:
        if merged and e.start < merged[-1].end:
            continue
        merged.append(e)
    return merged


async def clean_text(text: str, llm_classify_dates=None) -> dict:
    """Главная функция. llm_classify_dates: async fn(text, [Entity]) -> ['birth'|'keep', ...]
    для спорных дат; None -> спорные даты вычищаются (безопасный дефолт)."""
    entities = _find_regex_entities(text) + _find_ner_entities(text)

    dates = _classify_dates_heuristic(text)
    unsure = [e for e, verdict in dates if verdict == "unsure"]
    verdicts = {id(e): v for e, v in dates}
    if unsure and llm_classify_dates is not None:
        llm_out = await llm_classify_dates(text, unsure)
        for e, v in zip(unsure, llm_out):
            verdicts[id(e)] = v
    for e, _ in dates:
        v = verdicts[id(e)]
        if v == "unsure":
            v = "birth"  # сомневаемся -> вычищаем (лучше ложное срабатывание, чем утечка)
        if v == "birth":
            entities.append(e)

    entities = _merge(entities)

    # замена с конца, чтобы не сбить offsets; одинаковое значение -> один плейсхолдер
    counters: dict[str, int] = {}
    value_ph: dict[tuple, str] = {}
    removed = []
    out = text
    for e in sorted(entities, key=lambda x: -x.start):
        key = (e.kind, e.value.lower())
        if key not in value_ph:
            counters[e.kind] = counters.get(e.kind, 0) + 1
            value_ph[key] = f"[{PLACEHOLDER_RU[e.kind]}_{counters[e.kind]}]"
        out = out[:e.start] + value_ph[key] + out[e.end:]
        removed.append({"type": e.kind, "value": e.value, "placeholder": value_ph[key]})
    removed.reverse()
    return {"cleaned_text": out, "removed": removed, "entities_found": len(removed)}


def apply_extra_removals(cleaned: str, removed: list[dict], extra: list[dict]) -> tuple[str, list[dict]]:
    """Вычищает сущности, найденные LLM-аудитором вторым проходом."""
    counters: dict[str, int] = {}
    for r in removed:  # продолжить нумерацию плейсхолдеров
        k = r["placeholder"].strip("[]").rsplit("_", 1)
        counters[k[0]] = max(counters.get(k[0], 0), int(k[1]))
    for e in extra:
        kind = e["type"] if e["type"] in PLACEHOLDER_RU else "org"
        name = PLACEHOLDER_RU[kind]
        counters[name] = counters.get(name, 0) + 1
        ph = f"[{name}_{counters[name]}]"
        cleaned = cleaned.replace(e["value"], ph)
        removed.append({"type": kind, "value": e["value"], "placeholder": ph, "layer": "llm_audit"})
    return cleaned, removed
