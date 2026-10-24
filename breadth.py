"""Marktbreite-Test: S&P 500 (Einzelaktien -> Ampel-Kennzahlen).
Holt Kurse bei Yahoo, rechnet Trendanteil, neue Signale, Trendbrüche,
überkauft/überverkauft und schreibt data/breadth_sp500.json.
Läuft exit 1, wenn zu viele Aktien fehlen (= Test nicht bestanden)."""
import io, json, os, sys, time, datetime as dt
import pandas as pd
import requests
import yfinance as yf

WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT = "data/breadth_sp500.json"
MIN_QUOTE = 0.90  # mindestens 90 % der Aktien müssen geladen werden


def get_tickers():
    r = requests.get(WIKI, headers={"User-Agent": "Mozilla/5.0 (breadth-test)"}, timeout=30)
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.text))[0]
    return sorted({str(s).strip().replace(".", "-") for s in df["Symbol"]})


def download_closes(tickers, batch=100):
    parts = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(3):
            try:
                d = yf.download(chunk, period="2y", interval="1d", auto_adjust=True,
                                progress=False, threads=True)
                c = d["Close"]
                if isinstance(c, pd.Series):
                    c = c.to_frame(chunk[0])
                parts.append(c)
                break
            except Exception as e:
                print(f"Batch {i}: Versuch {attempt + 1} fehlgeschlagen: {e}")
                time.sleep(5 * (attempt + 1))
        time.sleep(2)
    closes = pd.concat(parts, axis=1)
    closes = closes.loc[:, ~closes.columns.duplicated()]
    return closes.dropna(axis=1, how="all").sort_index()


def compute(closes):
    sma50 = closes.rolling(50).mean()
    sma200 = closes.rolling(200).mean()
    valid = sma200.notna() & closes.notna()
    signal = (closes > sma50) & (sma50 > sma200) & valid
    prev = signal.shift(1, fill_value=False)
    new_sig = signal & ~prev
    broken = ~signal & prev & valid

    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)

    n = valid.sum(axis=1).replace(0, float("nan"))
    score = signal.sum(axis=1) / n * 100
    over = ((rsi > 70) & valid).sum(axis=1) / n * 100
    under = ((rsi < 30) & valid).sum(axis=1) / n * 100
    above50 = ((closes > sma50) & valid).sum(axis=1) / n * 100

    hist = [{"date": d.strftime("%Y-%m-%d"), "score": round(float(v), 1)}
            for d, v in score.dropna().tail(30).items()]
    return {
        "score": round(float(score.iloc[-1]), 1),
        "ueber_50_tage_pct": round(float(above50.iloc[-1]), 1),
        "neue_trendsignale": int(new_sig.iloc[-1].sum()),
        "trendbrueche": int(broken.iloc[-1].sum()),
        "ueberkauft_pct": round(float(over.iloc[-1]), 1),
        "ueberverkauft_pct": round(float(under.iloc[-1]), 1),
        "aktien_bewertet": int(valid.iloc[-1].sum()),
        "verlauf": hist,
    }


def main():
    tickers = get_tickers()
    print(f"Titel in Liste: {len(tickers)}")
    closes = download_closes(tickers)
    loaded = closes.shape[1]
    failed = sorted(set(tickers) - set(closes.columns))
    print(f"Geladen: {loaded} von {len(tickers)}; fehlend: {len(failed)}")
    res = compute(closes)
    res.update({
        "universum": "S&P 500",
        "stand_kurse": closes.index[-1].strftime("%Y-%m-%d"),
        "erzeugt_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "angefragt": len(tickers),
        "geladen": loaded,
        "fehlend": failed[:50],
        "definition": "Trendsignal = Kurs > 50-Tage-Linie und 50-Tage-Linie > 200-Tage-Linie. "
                      "Score = Anteil der Aktien mit Trendsignal in %. Überkauft = RSI(14) > 70, überverkauft = RSI(14) < 30.",
    })
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("verlauf", "fehlend")}, ensure_ascii=False, indent=1))
    if loaded < MIN_QUOTE * len(tickers):
        print("TEST NICHT BESTANDEN: zu wenige Aktien geladen.")
        sys.exit(1)
    print("TEST BESTANDEN.")


if __name__ == "__main__":
    main()
