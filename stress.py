"""Markt-Stress-Indikatoren -> data/stress.json (Yahoo + FRED ohne Schlüssel)."""
import io, json, os, datetime as dt
import pandas as pd, requests, yfinance as yf

UA = {"User-Agent": "Mozilla/5.0 (stress)"}


def fred(sid):
    key = (os.environ.get("FRED_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("FRED_API_KEY fehlt im Lauf (Secret wurde nicht uebergeben)")
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": sid, "api_key": key, "file_type": "json", "sort_order": "desc", "limit": 20},
                     headers=UA, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"FRED-API HTTP {r.status_code}")
    obs = [o for o in r.json()["observations"] if o["value"] != "."]
    return pd.Series({pd.to_datetime(o["date"]): float(o["value"]) for o in obs}).sort_index()


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



GRUPPEN = [
    ("USA", True, [
        ("^GSPC", "S&P 500", "Leitbarometer, Basis des S&P-Tech-ETFs und großer Teil des FTSE All-World."),
        ("^NDX", "Nasdaq 100", "Tech-Schwergewichte, geben oft die Richtung für Tech-ETFs vor."),
        ("^SOX", "Halbleiter (SOX)", "Chiphersteller als Taktgeber für Nasdaq und Tech."),
        ("^DJI", "Dow Jones", "Alte Industrie- und Finanzwerte, zeigt die Breite abseits von Tech."),
        ("^RUT", "Russell 2000", "Kleinwerte, Gradmesser für Risikobereitschaft und Zinssorgen.")]),
    ("Europa", True, [
        ("^STOXX", "Stoxx Europe 600", "Breites Europa, Heimat deiner Dividenden-ETFs."),
        ("^STOXX50E", "Euro Stoxx 50", "Europäische Schwergewichte."),
        ("^GDAXI", "DAX", "Deutschland, exportstark, reagiert früh auf Weltkonjunktur."),
        ("^FTSE", "FTSE 100", "Großbritannien, rohstofflastig.")]),
    ("Asien", True, [
        ("^N225", "Nikkei 225", "Japan, großer Anteil im FTSE All-World, stark vom Yen beeinflusst."),
        ("^KS11", "KOSPI (Südkorea)", "Speicherchips (u. a. Samsung, SK Hynix), Lieferkette für Tech und KI."),
        ("^TWII", "Taiwan (TAIEX)", "TSMC, Herzstück der Chip-Lieferkette."),
        ("^HSI", "Hang Seng", "China und Hongkong, Teil des FTSE All-World."),
        ("^NSEI", "Nifty 50 (Indien)", "Wachsender Anteil in Welt-ETFs.")]),
    ("Treiber", False, [
        ("DX-Y.NYB", "US-Dollar-Index", "Starker Dollar belastet oft Aktien und Schwellenländer."),
        ("^TNX", "US-10J-Rendite", "Steigende Zinsen drücken vor allem Tech-Bewertungen."),
        ("EURUSD=X", "Euro/Dollar", "Wirkt direkt auf deine Euro-Rendite bei US-Aktien."),
        ("USDJPY=X", "Dollar/Yen", "Ein schnell steigender Yen kann Verkäufe an den Märkten auslösen."),
        ("CL=F", "Rohöl (WTI)", "Inflations- und Konjunkturbarometer."),
        ("GC=F", "Gold", "Sicherer Hafen."),
        ("HG=F", "Kupfer", "Konjunkturbarometer."),
        ("BTC-USD", "Bitcoin", "Risikoappetit der Anleger.")]),
]


def index_stats(c, bp=False):
    c = c.dropna()
    last, prev = float(c.iloc[-1]), float(c.iloc[-2])
    s50 = c.rolling(50).mean().iloc[-1]
    s200 = c.rolling(200).mean().iloc[-1]
    trend = None
    if pd.notna(s200):
        trend = "auf" if last > s50 > s200 else ("ab" if last < s50 < s200 else "gemischt")
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    l = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    rsi = 100 - 100 / (1 + g / l) if l else 100.0
    ch = lambda a, b: (a - b) * 100 if bp else (a / b - 1) * 100
    return {"kurs": round(last, 2), "tag": round(ch(last, prev), 2), "trend": trend,
            "abstand_hoch_pct": round((last / float(c.tail(252).max()) - 1) * 100, 1),
            "rsi": round(float(rsi), 0),
            "m1": round(ch(last, float(c.iloc[-22])), 1) if len(c) > 22 else None,
            "datum": c.index[-1].strftime("%Y-%m-%d"), "einheit": "bp" if bp else "%"}


def indices():
    syms = [x[0] for g in GRUPPEN for x in g[2]]
    cl = None
    for _ in range(3):
        try:
            cl = yf.download(syms, period="2y", interval="1d", auto_adjust=True, progress=False, threads=True)["Close"]
            break
        except Exception as e:
            print("Indizes-Download:", e)
    out, cnt = [], {"auf": 0, "gemischt": 0, "ab": 0}
    for name, rate, items in GRUPPEN:
        rows = []
        for sym, nm, why in items:
            try:
                st = index_stats(cl[sym], bp=(sym == "^TNX"))
                if not rate:
                    st["trend"] = None
                elif st["trend"]:
                    cnt[st["trend"]] += 1
                st.update({"symbol": sym, "name": nm, "warum": why})
                rows.append(st)
            except Exception as e:
                rows.append({"symbol": sym, "name": nm, "warum": why, "fehler": str(e)[:80]})
                print("FEHLER", sym, e)
        out.append({"name": name, "items": rows})
    json.dump({"gruppen": out, "zusammenfassung": {**cnt, "gesamt": sum(cnt.values())},
               "erzeugt_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")},
              open("data/indices.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("Indizes:", cnt)


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
            row.update({"wert": None, "stufe": None, "datum": None, "fehler": str(e)[:160]})
            print("FEHLER", name, e)
        out.append(row)
    score = round(pts / (2 * n) * 100, 1) if n else None
    os.makedirs("data", exist_ok=True)
    json.dump({"score": score, "indikatoren": out, "geladen": n, "gesamt": len(IND),
               "erzeugt_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")},
              open("data/stress.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    try:
        hp = "data/stress_history.json"
        hist = json.load(open(hp, encoding="utf-8")) if os.path.exists(hp) else []
        today = dt.datetime.utcnow().strftime("%Y-%m-%d")
        hist = [x for x in hist if x.get("date") != today] + [{"date": today, "score": score, "geladen": n}]
        json.dump(hist[-400:], open(hp, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception as e:
        print("Verlauf nicht gespeichert:", e)
    print("Stress-Score:", score, f"({n} von {len(IND)} Indikatoren)")
    try:
        indices()
    except Exception as e:
        print("Weltmärkte fehlgeschlagen:", e)


if __name__ == "__main__":
    main()
