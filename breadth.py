"""Marktbreite für mehrere Märkte. Schreibt data/breadth_<id>.json und data/index.json."""
import io, json, os, sys, time, datetime as dt
import pandas as pd, requests, yfinance as yf

UA = {"User-Agent": "Mozilla/5.0 (breadth)"}
MIN_QUOTE = 0.85


def wiki(url, names, lo, hi):
    t_all = pd.read_html(io.StringIO(requests.get(url, headers=UA, timeout=30).text))
    seen = []
    for t in t_all:
        for c in t.columns:
            n = str(c[-1] if isinstance(c, tuple) else c).strip().lower()
            if any(k in n for k in names):
                seen.append(f"{n}:{len(t)}")
                if lo <= len(t) <= hi:
                    return [str(x).strip() for x in t[c].dropna()]
    raise ValueError("keine passende Tabelle (gefunden: " + ", ".join(seen[:6]) + ")")


UNIVERSES = [
    ("sp500", "S&P 500", lambda: [s.replace(".", "-") for s in wiki(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ("symbol",), 450, 520)]),
    ("ndx", "Nasdaq 100", lambda: wiki("https://en.wikipedia.org/wiki/Nasdaq-100", ("ticker", "symbol"), 90, 110)),
    ("dax", "DAX 40", lambda: [s if "." in s else s + ".DE" for s in wiki(
        "https://en.wikipedia.org/wiki/DAX", ("ticker",), 38, 45)]),
    ("dow", "Dow Jones", lambda: wiki("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", ("symbol",), 28, 32)),
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


def main():
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    lists, errors = {}, []
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
        res = compute(sub)
        res.update({"universum": name, "stand_kurse": sub.index[-1].strftime("%Y-%m-%d"), "erzeugt_utc": now,
                    "angefragt": len(tk), "geladen": len(cols),
                    "fehlend": sorted(set(tk) - set(cols))[:50],
                    "definition": "Trendsignal = Kurs > 50-Tage-Linie und 50-Tage-Linie > 200-Tage-Linie. "
                                  "Score = Anteil der Aktien mit Trendsignal in %. Überkauft = RSI(14) > 70, überverkauft = RSI(14) < 30."})
        json.dump(res, open(f"data/breadth_{uid}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        markets.append({"id": uid, "name": name})
        print(name, {k: v for k, v in res.items() if k in ("score", "neue_trendsignale", "trendbrueche")})
    json.dump({"markets": markets, "fehler": errors, "erzeugt_utc": now},
              open("data/index.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not any(m["id"] == "sp500" for m in markets):
        print("TEST NICHT BESTANDEN: S&P 500 fehlt.")
        sys.exit(1)
    print("OK:", [m["name"] for m in markets], "| Fehler:", errors)


if __name__ == "__main__":
    main()
