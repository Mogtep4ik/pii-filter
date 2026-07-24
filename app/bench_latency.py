"""Замер задержки фильтра по конфигурациям.

Зачем: у фильтра три слоя, и каждый следующий добавляет и точность, и время.
Чтобы выбрать режим для гейтвея, нужно видеть цену каждого слоя в цифрах,
а не среднее по всему пайплайну.

Конфигурации:
  regex      — только регексы (машинные форматы: телефон, email, ИНН, счёт...);
  regex+ner  — + Natasha (русские ФИО и организации) — режим `fast` в API;
  full       — + LLM-аудит вторым проходом — режим `full` в API.

Замеряем медиану, p95 и максимум: среднее по такой выборке малоинформативно,
потому что документы сильно разной длины, а LLM даёт длинный хвост.

Запуск:
    python -m app.bench_latency              # regex и regex+ner (без Ollama)
    python -m app.bench_latency --llm        # + полный режим (нужна Ollama)
    python -m app.bench_latency --llm -n 20  # ограничить выборку для full
"""
import argparse
import asyncio
import json
import pathlib
import statistics
import time

from .pii import clean_text

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def percentile(values: list[float], q: float) -> float:
    """q-й процентиль (0..1) по отсортированной выборке, ближайший ранг."""
    s = sorted(values)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def summarize(name: str, times: list[float], docs: int) -> dict:
    return {
        "config": name,
        "docs": docs,
        "median_ms": round(statistics.median(times), 1),
        "p95_ms": round(percentile(times, 0.95), 1),
        "max_ms": round(max(times), 1),
    }


async def measure(docs: list[dict], use_ner: bool, llm_fn=None, audit=None) -> list[float]:
    times = []
    for doc in docs:
        t0 = time.perf_counter()
        res = await clean_text(doc["text"], llm_classify_dates=llm_fn, use_ner=use_ner)
        if audit:
            extra = await audit[0](res["cleaned_text"])
            audit[1](res["cleaned_text"], res["removed"], extra)
        times.append((time.perf_counter() - t0) * 1000)
    return times


async def main(use_llm: bool, llm_docs: int) -> None:
    docs = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
    rows = []

    # прогрев: первый вызов тянет модели natasha в память, иначе он утянет max вверх
    await clean_text(docs[0]["text"])

    print(f"Корпус: {len(docs)} документов\n")

    for name, use_ner in (("regex", False), ("regex+ner", True)):
        times = await measure(docs, use_ner=use_ner)
        row = summarize(name, times, len(docs))
        rows.append(row)
        print(f"{name:12} медиана {row['median_ms']:>8} мс   p95 {row['p95_ms']:>8} мс   "
              f"max {row['max_ms']:>8} мс")

    if use_llm:
        from .llm import classify_dates, audit_residual_pii
        from .pii import apply_extra_removals
        subset = docs[:llm_docs]
        print(f"\nПолный режим: {len(subset)} док. (LLM медленная, полный корпус не нужен)")
        times = await measure(subset, use_ner=True,
                              llm_fn=classify_dates,
                              audit=(audit_residual_pii, apply_extra_removals))
        row = summarize("full (+LLM)", times, len(subset))
        rows.append(row)
        print(f"{'full (+LLM)':12} медиана {row['median_ms']:>8} мс   p95 {row['p95_ms']:>8} мс   "
              f"max {row['max_ms']:>8} мс")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latency.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    base = rows[0]["median_ms"]
    lines = ["# Задержка по конфигурациям", "",
             f"Корпус: {len(docs)} размеченных документов. Метрика — время очистки одного документа.",
             "Медиана и p95 вместо среднего: документы разной длины, у LLM длинный хвост.", "",
             "| Конфигурация | Документов | Медиана | p95 | Максимум | Во сколько раз медленнее regex |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['docs']} | {r['median_ms']} мс | {r['p95_ms']} мс | "
                     f"{r['max_ms']} мс | ×{r['median_ms'] / base:.1f} |")
    (RESULTS / "latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {RESULTS / 'latency.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="замерить и полный режим (нужна Ollama)")
    ap.add_argument("-n", type=int, default=15, help="сколько документов гнать через LLM")
    args = ap.parse_args()
    asyncio.run(main(args.llm, args.n))
