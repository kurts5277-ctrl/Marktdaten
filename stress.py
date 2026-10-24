"""Markt-Stress-Indikatoren -> data/stress.json (Yahoo + FRED ohne Schlüssel)."""
import io, json, os, datetime as dt
import pandas as pd, requests, yfinance as yf

UA = {"User-Agent": "Mozilla/5.0 (stress)"}


def fred(sid):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", headers=UA, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def yahoo(sym):
    c = yf.download(sym, period="1y", interval="1d", auto_adjust=True, progress=False)["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna()


def hi(y, r):  # hoher Wert = schlecht
    return lambda v: 0 if v < y else (1 if v < r else 2)


def lo(y, r):  # niedriger Wert = schlecht
    return lambda v: 0 if v > y else (1 if v > r else 2)


def vix_ratio():
    a, b = yahoo("^VIX"), yahoo("^VIX3M")
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    return j["a"] / j["b"]


def drawdown():
    c = yahoo("^GSPC")
    return (c / c.max() - 1) * 100


IND = [
    ("VIX (Angstindex)", "Punkte", lambda: yahoo("^VIX"), hi(20, 30), "Yahoo", "unter 20 ruhig, 20–30 erhöht, ab 30 Stress"),
    ("VIX / VIX3M (Kurvenform)", "Verhältnis", vix_ratio, hi(0.9, 1.0), "Yahoo", "ab 1,0 = Backwardation (akuter Stress)"),
    ("Kreditspread Hochzins (HY)", "%-Pkt.", lambda: fred("BAMLH0A0HYM2"), hi(4, 6), "FRED", "unter 4 ruhig, 4–6 erhöht, ab 6 Stress"),
    ("Kreditspread Investment Grade", "%-Pkt.", lambda: fred("BAMLC0A0CM"), hi(1.3, 2.0), "FRED", "unter 1,3 ruhig, 1,3–2,0 erhöht, ab 2,0 Stress"),
    ("Zinskurve 10J minus 2J", "%-Pkt.", lambda: fred("T10Y2Y"), lo(0.5, 0.0), "FRED", "über 0,5 ruhig, 0–0,5 erhöht, unter 0 invers"),
    ("Finanzstress-Index (St. Louis Fed)", "Index", lambda: fred("STLFSI4"), hi(0, 1), "FRED", "unter 0 = unterdurchschnittlicher Stress, ab 1 hoch"),
    ("S&P 500 Abstand zum 52-Wochen-Hoch", "%", drawdown, lo(-5, -10), "Yahoo", "bis -5 % ruhig, -5 bis -10 % erhöht, darunter Korrektur"),
]


def main():
    out, pts, n = [], 0, 0
    for name, unit, fn, lvl, src, rule in IND:
        row = {"name": name, "einheit": unit, "quelle": src, "regel": rule}
        try:
            s = fn()
            v = float(s.iloc[-1])
            row.update({"wert": round(v, 2), "stufe": lvl(v), "datum": s.index[-1].strftime("%Y-%m-%d")})
            pts += row["stufe"]
            n += 1
        except Exception as e:
            row.update({"wert": None, "stufe": None, "datum": None, "fehler": str(e)[:120]})
            print("FEHLER", name, e)
        out.append(row)
    score = round(pts / (2 * n) * 100, 1) if n else None
    os.makedirs("data", exist_ok=True)
    json.dump({"score": score, "indikatoren": out, "geladen": n, "gesamt": len(IND),
               "erzeugt_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")},
              open("data/stress.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("Stress-Score:", score, f"({n} von {len(IND)} Indikatoren)")


if __name__ == "__main__":
    main()
