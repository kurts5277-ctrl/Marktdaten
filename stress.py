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
    dup = abs(last - prev) < 1e-9
    return {"kurs": round(last, 2), "tag": None if dup else round(ch(last, prev), 2),
            "warn": "Kurs identisch zum Vortag, Tagesänderung nicht verlässlich" if dup else None, "trend": trend,
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
    return {"cnt": cnt, "rows": {r["symbol"]: r for g in out for r in g["items"]}}



ETFS = [("VWCE", "Vanguard FTSE All-World UCITS ETF (Acc)"), ("IQGA", "Invesco Global Enhanced Equity UCITS ETF"),
        ("JPGL", "JPMorgan Global Equity Multi-Factor UCITS ETF"), ("DFEN", "VanEck Defense UCITS ETF"),
        ("QDVE", "iShares S&P 500 Information Technology Sector UCITS ETF"),
        ("LDGA", "L&G Quality Dividends"), ("L0CK", "iShares Digital Security UCITS ETF")]


def etfs():
    rows = []
    for k, name in ETFS:
        row = {"kuerzel": k, "name": name}
        for sf in (".DE", ".L", ".MI", ".AS", ""):
            try:
                c = yahoo(k + sf)
                if len(c) > 60:
                    row.update(index_stats(c))
                    row["symbol"] = k + sf
                    break
            except Exception:
                pass
        if "kurs" not in row:
            row["fehler"] = "Kürzel bei Yahoo nicht gefunden"
            print("FEHLER ETF", k)
        rows.append(row)
    json.dump({"etfs": rows, "erzeugt_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")},
              open("data/etfs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def push(text):
    t = (os.environ.get("NTFY_TOPIC") or "").strip()
    if not t:
        return
    try:
        requests.post("https://ntfy.sh/" + t, data=text.encode("utf-8"), headers={"Title": "Boersenampel"}, timeout=15)
    except Exception as e:
        print("Push fehlgeschlagen:", type(e).__name__)


NAMEN = ["Gefahr: breiter Abverkauf", "Vorsicht", "Gemischtes Bild", "Überwiegend freundlich", "Rückenwind"]


def gesamt(stress_score, ix):
    try:
        b = json.load(open("data/breadth_sp500.json", encoding="utf-8"))["score"]
    except Exception:
        b = None
    z = ix["cnt"] if ix else {"auf": 0, "gemischt": 0, "ab": 0}
    tot = sum(z.values())
    w = (z["auf"] + 0.5 * z["gemischt"]) / tot * 100 if tot else None
    ruhe = None if stress_score is None else 100 - stress_score
    teile = []

    def teil(name, v, worte):
        if v is None:
            return None
        i = 0 if v < 40 else (1 if v < 60 else 2)
        teile.append({"name": name, "urteil": worte[i], "farbe": ["rot", "gelb", "gruen"][i], "wert": round(v)})
        return v

    vals = [x for x in (teil("Breite S&P 500", b, ["schwach", "gemischt", "breit"]),
                        teil("Stress", ruhe, ["unter Druck", "erhöht", "ruhig"]),
                        teil("Weltmärkte", w, ["schwach", "gemischt", "stark"])) if x is not None]
    if not vals:
        return
    sc = sum(vals) / len(vals)
    stufe = 0 if sc < 20 else 1 if sc < 40 else 2 if sc < 60 else 3 if sc < 80 else 4
    ein = []
    if stress_score is not None:
        ein.append("Kredit- und Angstmärkte sind " + ("ruhig." if stress_score < 33 else "angespannt." if stress_score < 66 else "stark unter Druck."))
    if b is not None:
        ein.append("Im S&P 500 läuft " + ("nur eine Minderheit" if b < 40 else "etwa die Hälfte" if b < 60 else "die Mehrheit") + f" der Aktien im Aufwärtstrend ({b:.0f} %).")
    if tot:
        ein.append(f"{z['auf']} von {tot} wichtigen Aktienindizes weltweit sind im Aufwärtstrend, {z['ab']} im Abwärtstrend.")
    rows = ix["rows"] if ix else {}
    sox, ndx = rows.get("^SOX", {}).get("abstand_hoch_pct"), rows.get("^NDX", {}).get("abstand_hoch_pct")
    tech_schwach = (sox is not None and sox < -15) or (ndx is not None and ndx < -10)
    if sox is not None and ndx is not None:
        ein.append(f"Halbleiter liegen {abs(sox):.0f} % und der Nasdaq 100 {abs(ndx):.0f} % unter ihrem Hoch.")
    if stress_score is not None and stress_score < 33 and b is not None and b < 40:
        ein.append("Meine Einordnung: Die Teile widersprechen sich. Der Druck sitzt in einzelnen Bereichen, nicht im ganzen Markt." + (" Er kommt vor allem aus Tech und Chips." if tech_schwach else ""))
    if tech_schwach:
        ein.append("Das trifft deinen S&P-Tech-ETF (iShares S&P 500 Information Technology) direkt, deinen Welt-ETF (Vanguard FTSE All-World) nur teilweise, weil Tech dort nur ein Teil ist.")
    else:
        ein.append("Dein Welt-ETF (Vanguard FTSE All-World) folgt im Wesentlichen diesem Gesamtbild.")
    ein.append("Das ist ein Lagebild, kein Kauf- oder Verkaufssignal. Dein 20-Jahre-Sparplan hängt nicht an dieser Ampel.")
    stress_stufe = None if stress_score is None else (0 if stress_score < 33 else 1 if stress_score < 66 else 2)
    prev = None
    try:
        prev = json.load(open("data/gesamt.json", encoding="utf-8"))
    except Exception:
        pass
    out = {"stufe": stufe, "name": NAMEN[stufe], "score": round(sc, 1), "teile": teile, "einordnung": ein,
           "stress_stufe": stress_stufe, "erzeugt_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")}
    json.dump(out, open("data/gesamt.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if prev and prev.get("stufe") != stufe:
        push(f"Gesamtbild: {prev.get('name')} → {NAMEN[stufe]} ({sc:.0f} von 100)")
    sn = ["ruhig", "erhöht", "hoch"]
    if prev and prev.get("stress_stufe") is not None and stress_stufe is not None and prev["stress_stufe"] != stress_stufe:
        push(f"Markt-Stress: {sn[prev['stress_stufe']]} → {sn[stress_stufe]} ({stress_score:.0f} von 100)")
    if not os.path.exists("data/ntfy_ok.txt") and (os.environ.get("NTFY_TOPIC") or "").strip():
        push("Verbindung steht. Du bekommst eine Nachricht, wenn Gesamtbild oder Markt-Stress die Stufe wechseln.")
        open("data/ntfy_ok.txt", "w").write("ok")


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
    ix = None
    try:
        ix = indices()
    except Exception as e:
        print("Weltmärkte fehlgeschlagen:", e)
    try:
        etfs()
    except Exception as e:
        print("ETFs fehlgeschlagen:", e)
    try:
        gesamt(score, ix)
    except Exception as e:
        print("Gesamtbild fehlgeschlagen:", e)


if __name__ == "__main__":
    main()
