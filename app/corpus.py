"""Генератор размеченного корпуса: фрагменты русских документов с известными PII.

Каждый документ — текст + разметка: какие значения ДОЛЖНЫ быть вычищены (pii)
и какие ловушки ДОЛЖНЫ остаться (keep): даты подписания/сроки, суммы, номера законов.
На этом корпусе честно считаем recall (ничего не пропустили) и precision (не вычистили лишнее).
"""
import json, random, pathlib

random.seed(2026)

FIRST = ["Иван", "Пётр", "Сергей", "Анна", "Мария", "Дмитрий", "Ольга", "Алексей", "Екатерина", "Николай"]
LAST = ["Иванов", "Петров", "Сидоров", "Кузнецов", "Смирнов", "Васильев", "Морозов", "Волков", "Соколов", "Козлов"]
MID = ["Иванович", "Петрович", "Сергеевич", "Николаевич", "Андреевна", "Владимировна", "Алексеевна"]
ORGS = ["ООО «Ромашка»", "АО «ТехноСервис»", "ООО «СтройГарант»", "ПАО «ЭнергоСбыт»", "ООО «Вектор Плюс»", "АО «Логистика Центр»"]
BRANCHES = ["филиал «Северо-Западный»", "Уральский филиал", "филиал в г. Казани", "обособленное подразделение «Юг»"]
CITIES = ["Москва", "Санкт-Петербург", "Казань", "Екатеринбург", "Новосибирск"]

def fio():
    l, f, m = random.choice(LAST), random.choice(FIRST), random.choice(MID)
    if f in ("Анна", "Мария", "Ольга", "Екатерина") and m.endswith("ич"):
        m = m[:-2] + "на"
    if f not in ("Анна", "Мария", "Ольга", "Екатерина") and m.endswith("на"):
        m = m[:-2] + "ич"
    if f in ("Анна", "Мария", "Ольга", "Екатерина"):
        l += "а"
    return f"{l} {f} {m}"

def phone():
    return random.choice(["+7 ", "8 "]) + f"({random.randint(900,999)}) {random.randint(100,999)}-{random.randint(10,99)}-{random.randint(10,99)}"

def email(name):
    translit = {"а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","и":"i","к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f"}
    base = "".join(translit.get(c, "") for c in name.split()[0].lower())[:8] or "user"
    return f"{base}{random.randint(1,99)}@{random.choice(['mail.ru','yandex.ru','gmail.com','company.ru'])}"

def inn():   return "".join(str(random.randint(0, 9)) for _ in range(random.choice([10, 12])))
def snils(): return f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(100,999)} {random.randint(10,99)}"
def acct():  return "407028" + "".join(str(random.randint(0, 9)) for _ in range(14))
def d(y1=2020, y2=2026): return f"{random.randint(1,28):02d}.{random.randint(1,12):02d}.{random.randint(y1,y2)}"
def bday():  return d(1960, 2003)
def money(): return f"{random.randrange(100, 9999)} {random.choice(['000', '500'])} руб."

# Шаблоны: {слоты}. pii-слоты вычищаем, keep-слоты должны уцелеть.
TEMPLATES = [
    ("Договор № {keep_num} от {keep_date} заключён между {org1}, в лице директора {fio1}, "
     "и {org2}, в лице представителя {fio2}, действующего на основании доверенности. "
     "Контактный телефон: {phone1}, e-mail: {email1}. Сумма договора составляет {keep_money}. "
     "Срок исполнения обязательств — до {keep_date2}."),
    ("Заявка принята от {fio1} (дата рождения {bday1}, СНИЛС {snils1}). "
     "Просим связаться по номеру {phone1} до {keep_date}. "
     "Организация: {org1}, {branch1}, ИНН {inn1}."),
    ("Акт составлен {keep_date} в г. {city1}. Присутствовали: {fio1} ({org1}) и {fio2} ({org2}). "
     "Оплата в размере {keep_money} должна поступить на счёт {acct1} не позднее {keep_date2}. "
     "Ответственный: {fio3}, тел. {phone1}."),
    ("Уведомляем, что {fio1}, {bday1} года рождения, паспорт выдан {keep_date}, "
     "назначен ответственным по договору № {keep_num}. Вопросы направляйте на {email1} "
     "или по телефону {phone1}. Согласно ст. 432 ГК РФ договор вступает в силу с {keep_date2}."),
    ("Победителем закупки признано {org1} ({branch1}), ИНН {inn1}. "
     "Контактное лицо: {fio1}, {email1}, {phone1}. "
     "Итоговая цена контракта: {keep_money}. Подписание запланировано на {keep_date}."),
    ("Служебная записка. Прошу предоставить {fio1} (таб. номер {keep_num}) отпуск с {keep_date} по {keep_date2}. "
     "Замещать будет {fio2}, внутренний телефон {phone1}. Согласовано с руководством {org1}."),
]

MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]

def bday_words():
    return f"{random.randint(1,28)} {random.choice(MONTHS)} {random.randint(1960,2003)} года"

def initials(full):
    l, f, m = full.split()
    return f"{l} {f[0]}.{m[0]}."

def phone_flat():
    return "8" + str(random.randint(9000000000, 9999999999))

HARD_TEMPLATES = [
    ("Исполнитель: {fio_init1} (полн.: {fio1}), моб. {phone_flat1}. "
     "Заказчик — компания {org_bare1}, договор от {keep_date}, цена {keep_money}."),
    ("Гражданин {fio1}, {bday_words1} рождения, зарегистрирован по адресу: г. {city1}, ул. Ленина, д. 5. "
     "Явиться до {keep_date}. Справки: {phone_flat1}."),
    ("Подписи сторон: {fio_init1} ({org1}) / {fio_init2} ({org_bare1}). "
     "Приложение № {keep_num} к договору от {keep_date}. Контакт: {email1}."),
]

ORGS_BARE = ["Ромашка", "ТехноСервис", "СтройГарант", "Вектор Плюс"]

def gen_doc(i):
    hard = random.random() < 0.35
    t = random.choice(HARD_TEMPLATES if hard else TEMPLATES)
    fios = [fio() for _ in range(3)]
    slots = {
        "fio1": fios[0], "fio2": fios[1], "fio3": fios[2],
        "org1": random.choice(ORGS), "org2": random.choice(ORGS),
        "branch1": random.choice(BRANCHES),
        "phone1": phone(), "email1": email(fios[0]),
        "inn1": inn(), "snils1": snils(), "acct1": acct(),
        "bday1": bday(), "city1": random.choice(CITIES),
        "keep_date": d(), "keep_date2": d(), "keep_num": str(random.randrange(100, 99999)),
        "keep_money": money(),
        "fio_init1": initials(fios[0]), "fio_init2": initials(fios[1]),
        "phone_flat1": phone_flat(), "bday_words1": bday_words(),
        "org_bare1": "компания " + random.choice(ORGS_BARE) if False else random.choice(ORGS_BARE),
    }
    text = t.format(**slots)
    pii, keep = [], []
    for k, v in slots.items():
        if ("{" + k + "}") not in t:
            continue
        entry = {"type": k.rstrip("123"), "value": v}
        (keep if k.startswith("keep_") else pii).append(entry)
    return {"id": i, "text": text, "pii": pii, "keep": keep}

if __name__ == "__main__":
    docs = [gen_doc(i) for i in range(1, 121)]
    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "corpus.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    n_pii = sum(len(x["pii"]) for x in docs)
    n_keep = sum(len(x["keep"]) for x in docs)
    print(f"Документов: {len(docs)}, PII-сущностей: {n_pii}, ловушек keep: {n_keep} -> {out}")
