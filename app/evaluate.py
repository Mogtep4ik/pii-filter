"""Оценка качества очистки на размеченном корпусе.

- recall (полнота): доля PII-значений, которых НЕ осталось в очищенном тексте.
  Это главная метрика — «чтобы не пропускал» (утечка хуже ложного срабатывания).
- keep-точность: доля «ловушек» (даты договора, суммы, номера), которые уцелели.
  Это прокси precision — фильтр не должен калечить полезный текст.
- latency: среднее/медиана/p95 на документ.

Запуск: python -m app.evaluate [--llm]  (--llm подключает Ollama для спорных дат)
"""
import argparse, asyncio, json, pathlib, statistics, time, collections

from .pii import clean_text

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def value_leaked(value: str, cleaned: str, vtype: str) -> bool:
    """PII считается утёкшим, если его значение (или его цифровая форма) осталось в тексте."""
    if value in cleaned:
        return True
    if vtype in ("phone", "snils", "inn", "account", "card", "passport"):
        return len(_digits(value)) >= 6 and _digits(value) in _digits(cleaned)
    if vtype == "fio":  # утечкой считаем и отдельно уцелевшую фамилию
        return value.split()[0] in cleaned
    return False


async def main(use_llm: bool):
    llm_fn = None
    audit = None
    if use_llm:
        from .llm import classify_dates, audit_residual_pii
        from .pii import apply_extra_removals
        llm_fn = classify_dates
        audit = (audit_residual_pii, apply_extra_removals)

    docs = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
    leaks, kept_traps, total_pii, total_traps = [], 0, 0, 0
    by_type_total, by_type_leaked = collections.Counter(), collections.Counter()
    times = []

    for doc in docs:
        t0 = time.perf_counter()
        res = await clean_text(doc["text"], llm_classify_dates=llm_fn)
        cleaned = res["cleaned_text"]
        if audit:
            extra = await audit[0](cleaned)
            cleaned, _ = audit[1](cleaned, res["removed"], extra)
        times.append((time.perf_counter() - t0) * 1000)
        for p in doc["pii"]:
            total_pii += 1
            by_type_total[p["type"]] += 1
            if value_leaked(p["value"], cleaned, p["type"]):
                leaks.append({"doc": doc["id"], **p})
                by_type_leaked[p["type"]] += 1
        for k in doc["keep"]:
            total_traps += 1
            if k["value"] in cleaned:
                kept_traps += 1

    recall = 1 - len(leaks) / total_pii
    keep_rate = kept_traps / total_traps
    report = {
        "mode": "with_llm" if use_llm else "heuristics_only",
        "docs": len(docs),
        "pii_total": total_pii,
        "pii_leaked": len(leaks),
        "recall": round(recall, 4),
        "keep_traps_total": total_traps,
        "keep_traps_survived": kept_traps,
        "keep_rate": round(keep_rate, 4),
        "recall_by_type": {t: round(1 - by_type_leaked[t] / by_type_total[t], 4) for t in by_type_total},
        "latency_ms": {
            "mean": round(statistics.mean(times), 1),
            "median": round(statistics.median(times), 1),
            "p95": round(sorted(times)[int(len(times) * 0.95)], 1),
        },
        "leaks_sample": leaks[:15],
    }
    out = DATA / f"eval_{report['mode']}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recall (вычищено PII): {recall:.1%}  ({len(leaks)} утечек из {total_pii})")
    print(f"Keep-rate (ловушки уцелели): {keep_rate:.1%}  ({kept_traps}/{total_traps})")
    print("Recall по типам:", {t: f"{1 - by_type_leaked[t]/by_type_total[t]:.0%}" for t in by_type_total})
    print(f"Latency: mean {report['latency_ms']['mean']} ms, p95 {report['latency_ms']['p95']} ms")
    print(f"Отчёт: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="подключить LLM для спорных дат")
    asyncio.run(main(ap.parse_args().llm))
