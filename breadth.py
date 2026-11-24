"""Marktbreite für mehrere Märkte. Schreibt data/breadth_<id>.json und data/index.json."""
import io, json, os, sys, time, datetime as dt
import pandas as pd, requests, yfinance as yf

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9"}
MIN_QUOTE = 0.85


def wiki(urls, names, lo, hi):
    urls = [urls] if isinstance(urls, str) else list(urls)
    diag = []
    for url in urls:
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            tabs = pd.read_html(io.StringIO(r.text))
        except Exception as e:
            diag.append(url.split("/")[2] + "/" + url.rstrip("/").rsplit("/", 1)[-1] + ": " + type(e).__name__)
            continue
        for t in tabs:
            for c in t.columns:
                n = str(c[-1] if isinstance(c, tuple) else c).strip().lower()
                if any(k in n for k in names) and lo <= len(t) <= hi:
                    return [str(x).strip() for x in t[c].dropna()]
        heads = [",".join(str(c[-1] if isinstance(c, tuple) else c)[:12] for c in list(t.columns)[:3]) + f"({len(t)})" for t in tabs[:6]]
        diag.append(url.split("/")[2] + ": keine passende Tabelle: " + " | ".join(heads))
    raise ValueError("keine passende Tabelle. " + " ; ".join(diag)[:260])


UNIVERSES = [
    ("sp500", "S&P 500", lambda: [s.replace(".", "-") for s in wiki(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ("symbol",), 450, 520)]),
    # Wikipedia führt die Nasdaq-100-Mitglieder nicht mehr als Tabelle; stockanalysis.com liefert sie serverseitig gerendert
    ("ndx", "Nasdaq 100", lambda: wiki(["https://stockanalysis.com/list/nasdaq-100-stocks/",
                                        "https://en.wikipedia.org/wiki/Nasdaq-100"], ("symbol", "ticker"), 95, 110)),
    ("dax", "DAX 40", lambda: [s if "." in s else s + ".DE" for s in wiki(
        "https://en.wikipedia.org/wiki/DAX", ("ticker",), 38, 45)]),
    ("dow", "Dow Jones", lambda: wiki(["https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies", "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"], ("symbol", "ticker"), 28, 32)),
]


def download_closes(tickers, batch=100):
    parts = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(3):
            try:
                c = yf.download(chunk, period="2y", interval="1d", auto_adjust=True,
                                progress=False, threads=True)["Close"]
                if isinstance(c, pd.Series):
                    c = c.to_frame(chunk[0])
                parts.append(c)
                break
            except Exception as e:
                print(f"Batch {i}: Versuch {attempt + 1} fehlgeschlagen: {e}")
                time.sleep(5 * (attempt + 1))
        time.sleep(2)
    closes = pd.concat(parts, axis=1)
    return closes.loc[:, ~closes.columns.duplicated()].dropna(axis=1, how="all").sort_index()


def sp500_sektoren():
    for t in pd.read_html(io.StringIO(requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30).text)):
        cols = {str(c[-1] if isinstance(c, tuple) else c).strip().lower(): c for c in t.columns}
        if "symbol" in cols and "gics sector" in cols and len(t) > 450:
            return {str(a).strip().replace(".", "-"): str(x).strip() for a, x in zip(t[cols["symbol"]], t[cols["gics sector"]])}
    return {}


def sector_breadth(closes, sek):
    sma50 = closes.rolling(50).mean()
    sma200 = closes.rolling(200).mean()
    valid = sma200.notna() & closes.notna()
    sig = (closes > sma50) & (sma50 > sma200) & valid
    out = []
    for name in sorted(set(sek.values())):
        cols = [c for c in closes.columns if sek.get(c) == name]
        v = valid[cols]
        if not cols or v.iloc[-1].sum() == 0:
            continue
        share = lambda i: round(float(sig[cols].iloc[i].sum() / max(v.iloc[i].sum(), 1) * 100), 1)
        out.append({"sektor": name, "n": int(v.iloc[-1].sum()), "anteil": share(-1), "vor_5_tagen": share(-6)})
    return sorted(out, key=lambda x: -x["anteil"])


def compute(closes):
    sma50 = closes.rolling(50).mean()
    sma200 = closes.rolling(200).mean()
    valid = sma200.notna() & closes.notna()
    signal = (closes > sma50) & (sma50 > sma200) & valid
    prev = signal.shift(1, fill_value=False)
    new_sig = signal & ~prev
    broken = ~signal & prev & valid
    d = closes.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    n = valid.sum(axis=1).replace(0, float("nan"))
    pct = lambda m: ((m & valid).sum(axis=1) / n * 100)
    score = signal.sum(axis=1) / n * 100
    hist = [{"date": x.strftime("%Y-%m-%d"), "score": round(float(v), 1)} for x, v in score.dropna().tail(30).items()]
    return {
        "score": round(float(score.iloc[-1]), 1),
        "ueber_50_tage_pct": round(float(pct(closes > sma50).iloc[-1]), 1),
        "ueber_200_tage_pct": round(float(pct(closes > sma200).iloc[-1]), 1),
        "neue_trendsignale": int(new_sig.iloc[-1].sum()),
        "trendbrueche": int(broken.iloc[-1].sum()),
        "ueberkauft_pct": round(float(pct(rsi > 70).iloc[-1]), 1),
        "ueberverkauft_pct": round(float(pct(rsi < 30).iloc[-1]), 1),
        "aktien_bewertet": int(valid.iloc[-1].sum()),
        "verlauf": hist,
    }


def obos_series(closes, win=50):
    """Bespoke-Methode: Anteil der Aktien mit Schlusskurs mehr als 1 Standardabweichung über/unter ihrem 50-Tage-Durchschnitt."""
    ma = closes.rolling(win).mean()
    sd = closes.rolling(win).std(ddof=0)
    z = (closes - ma) / sd.where(sd > 0)
    valid = z.notna()
    n = valid.sum(axis=1).astype(float)
    n = n.where(n > 0)
    out = pd.DataFrame({"ob": ((z > 1) & valid).sum(axis=1) / n * 100,
                        "os": ((z < -1) & valid).sum(axis=1) / n * 100, "n": n})
    return out[out["n"] >= max(20, 0.6 * closes.shape[1])]


def write_obos(uid, name, closes, now):
    res = obos_series(closes).tail(380)
    if len(res) < 60:
        raise RuntimeError("zu wenig Verlauf")
    last = res.iloc[-1]
    out = {"universum": name,
           "definition": "Überkauft = Schlusskurs mehr als 1 Standardabweichung über dem 50-Tage-Durchschnitt, überverkauft = mehr als 1 Standardabweichung darunter (Standardabweichung der letzten 50 Schlusskurse). Anteil der Aktien in %.",
           "stand_kurse": res.index[-1].strftime("%Y-%m-%d"), "erzeugt_utc": now, "aktien_bewertet": int(last["n"]),
           "ueberkauft_pct": round(float(last["ob"]), 1), "ueberverkauft_pct": round(float(last["os"]), 1),
           "verlauf": [{"date": d.strftime("%Y-%m-%d"), "ob": round(float(r.ob), 1), "os": round(float(r.os), 1), "n": int(r.n)}
                       for d, r in res.iterrows()]}
    json.dump(out, open(f"data/obos_{uid}.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))


INDICES = [("sp500", "S&P 500", "^GSPC"), ("ndx", "Nasdaq 100", "^NDX"), ("dax", "DAX 40", "^GDAXI"), ("dow", "Dow Jones", "^DJI")]
NEU = {"tag": 10, "woche": 4, "monat": 2}      # Kreuzung gilt als neu, wenn sie höchstens so viele Kerzen her ist
NAH = {"tag": 1.0, "woche": 2.0, "monat": 3.0}  # Abstand 50er/200er in %, ab dem eine Kreuzung "droht"


def ma_block(ser, key):
    ser = ser.dropna()
    if len(ser) < 205:
        return None
    s50, s200 = ser.rolling(50).mean(), ser.rolling(200).mean()
    diff = (s50 - s200).dropna()
    sign = diff.apply(lambda v: 1 if v > 0 else -1)
    changed = sign.ne(sign.shift()) & sign.shift().notna()
    kreuz = None
    if changed.any():
        idx = changed[changed].index[-1]
        kreuz = {"typ": "golden" if sign.loc[idx] > 0 else "death", "datum": idx.strftime("%Y-%m-%d"),
                 "bars_her": int(len(sign) - 1 - sign.index.get_loc(idx))}
        kreuz["neu"] = kreuz["bars_her"] <= NEU[key]
    last, a50, a200 = float(ser.iloc[-1]), float(s50.iloc[-1]), float(s200.iloc[-1])
    gap = (a50 - a200) / a200 * 100
    konv = len(diff) > 6 and abs(float(diff.iloc[-1])) < abs(float(diff.iloc[-6]))
    return {"stand": ser.index[-1].strftime("%Y-%m-%d"), "kurs": round(last, 2),
            "sma50": round(a50, 2), "sma200": round(a200, 2),
            "abstand50_pct": round((last / a50 - 1) * 100, 2), "abstand200_pct": round((last / a200 - 1) * 100, 2),
            "luecke_pct": round(gap, 2), "zustand": "golden" if gap > 0 else "death",
            "nah": bool(abs(gap) < NAH[key] and konv), "kreuzung": kreuz}


def index_history(ticker):
    df = yf.download(ticker, period="max", interval="1d", auto_adjust=False, progress=False)
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna().sort_index()


def ma_all(now):
    res, problems = {}, []
    for uid, name, tk in INDICES:
        try:
            d = index_history(tk)
            try:
                w, m = d.resample("W-FRI").last(), d.resample("ME").last()
            except ValueError:  # ältere pandas-Versionen
                w, m = d.resample("W-FRI").last(), d.resample("M").last()
            res[uid] = {"name": name, "ticker": tk, "tag": ma_block(d, "tag"), "woche": ma_block(w, "woche"), "monat": ma_block(m, "monat")}
        except Exception as e:
            problems.append(f"Trendlinien {name}: {str(e)[:60]}")
    if res:
        json.dump({"erzeugt_utc": now, "indizes": res}, open("data/ma.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    return problems



def main():
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    lists, errors, notes = {}, [], []
    for uid, name, fn in UNIVERSES:
        try:
            lists[uid] = (name, fn())
            print(f"{name}: {len(lists[uid][1])} Titel")
        except Exception as e:
            errors.append(f"{name}: Liste fehlgeschlagen ({e})")
            print("FEHLER", errors[-1])
    closes = download_closes(sorted({t for _, l in lists.values() for t in l}))
    os.makedirs("data", exist_ok=True)
    markets = []
    for uid, (name, tk) in lists.items():
        cols = [t for t in tk if t in closes.columns]
        print(f"{name}: geladen {len(cols)} von {len(tk)}")
        if len(cols) < MIN_QUOTE * len(tk):
            errors.append(f"{name}: nur {len(cols)} von {len(tk)} geladen (fehlt z. B. {', '.join(sorted(set(tk) - set(cols))[:4])})")
            continue
        sub = closes[cols].dropna(how="all").ffill(limit=5)
        exp = pd.Timestamp(dt.datetime.utcnow().date())
        if dt.datetime.utcnow().hour < 22:
            exp -= pd.Timedelta(days=1)
        while exp.weekday() >= 5:
            exp -= pd.Timedelta(days=1)
        if sub.index[-1] < exp:  # nur ein Hinweis, der Markt wird trotzdem gezeigt
            notes.append(f"{name}: Kurse enden am {sub.index[-1]:%d.%m.}, ein Handelstag älter als bei den anderen Märkten")
        res = compute(sub)
        try:
            write_obos(uid, name, sub, now)
        except Exception as e:
            notes.append(f"{name}: Marktstimmung nicht berechnet ({str(e)[:50]})")
        if uid == "sp500":
            try:
                sk = sp500_sektoren()
                if sk:
                    json.dump({"sektoren": sector_breadth(sub, sk), "erzeugt_utc": now},
                              open("data/sektoren.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            except Exception as e:
                errors.append(f"Sektoren: {str(e)[:60]}")
        res.update({"universum": name, "stand_kurse": sub.index[-1].strftime("%Y-%m-%d"), "erzeugt_utc": now,
                    "angefragt": len(tk), "geladen": len(cols),
                    "fehlend": sorted(set(tk) - set(cols))[:50],
                    "definition": "Trendsignal = Kurs > 50-Tage-Linie und 50-Tage-Linie > 200-Tage-Linie. "
                                  "Score = Anteil der Aktien mit Trendsignal in %. Überkauft = RSI(14) > 70, überverkauft = RSI(14) < 30."})
        json.dump(res, open(f"data/breadth_{uid}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        markets.append({"id": uid, "name": name})
        print(name, {k: v for k, v in res.items() if k in ("score", "neue_trendsignale", "trendbrueche")})
    notes += ma_all(now)
    json.dump({"markets": markets, "fehler": errors, "hinweise": notes, "erzeugt_utc": now},
              open("data/index.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not any(m["id"] == "sp500" for m in markets):
        print("TEST NICHT BESTANDEN: S&P 500 fehlt.")
        sys.exit(1)
    print("OK:", [m["name"] for m in markets], "| Fehler:", errors, "| Hinweise:", notes)


if __name__ == "__main__":
    main()
