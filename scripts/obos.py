"""Überkauft/Überverkauft im S&P 500 nach Bespoke-Methode.

Eine Aktie gilt als überkauft, wenn ihr Schlusskurs mehr als 1 Standardabweichung ÜBER ihrem 50-Tage-Durchschnitt liegt,
als überverkauft, wenn er mehr als 1 Standardabweichung DARUNTER liegt. Standardabweichung = der Schlusskurse der letzten 50 Tage.
Ausgabe: data/obos_sp500.json mit dem Anteil in % für jeden Handelstag (rund 1,5 Jahre Verlauf).
"""
import io, json, sys, datetime as dt
import pandas as pd

WIN = 50
HIST_DAYS = 380  # so viele Handelstage Verlauf werden gespeichert


def universe():
    """S&P-500-Mitglieder von Wikipedia (Yahoo-Schreibweise: Punkt -> Bindestrich)."""
    import requests
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0 (Marktdaten-Pipeline)"}, timeout=60)
    r.raise_for_status()
    t = pd.read_html(io.StringIO(r.text))[0]
    syms = [str(s).strip().replace(".", "-") for s in t["Symbol"].tolist()]
    if len(syms) < 400:
        raise RuntimeError(f"Mitgliederliste unvollständig: {len(syms)}")
    return syms


def download(tickers):
    import yfinance as yf
    parts = []
    for i in range(0, len(tickers), 100):  # in Paketen, schont das Limit von Yahoo
        chunk = tickers[i:i + 100]
        df = yf.download(chunk, period="2y", interval="1d", auto_adjust=False, progress=False, threads=True)
        close = df["Close"] if "Close" in df else df
        if isinstance(close, pd.Series):
            close = close.to_frame(chunk[0])
        parts.append(close)
    close = pd.concat(parts, axis=1)
    close = close.loc[:, ~close.columns.duplicated()].dropna(how="all")
    return close


def compute(close, win=WIN):
    """close: Tabelle Datum x Aktie. Rückgabe: Tabelle Datum -> ob, os, n (in %)."""
    ma = close.rolling(win).mean()
    sd = close.rolling(win).std(ddof=0)
    z = (close - ma) / sd
    z = z.where(sd > 0)
    valid = z.notna()
    n = valid.sum(axis=1).astype(float)
    n = n.where(n > 0)  # Tage ohne Daten -> NaN statt Division durch 0
    ob = ((z > 1) & valid).sum(axis=1) / n * 100
    os_ = ((z < -1) & valid).sum(axis=1) / n * 100
    out = pd.DataFrame({"ob": ob, "os": os_, "n": n})
    return out[out["n"] >= 300]  # nur Tage mit breiter Datenbasis


def main():
    tickers = universe()
    close = download(tickers)
    res = compute(close).tail(HIST_DAYS)
    if len(res) < 60:
        raise RuntimeError("zu wenig Verlauf berechnet")
    last = res.iloc[-1]
    out = {
        "universum": "S&P 500",
        "definition": "Überkauft = Schlusskurs mehr als 1 Standardabweichung über dem 50-Tage-Durchschnitt, überverkauft = mehr als 1 Standardabweichung darunter (Standardabweichung der letzten 50 Schlusskurse). Anteil der Aktien in %.",
        "stand_kurse": res.index[-1].strftime("%Y-%m-%d"),
        "erzeugt_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "aktien_bewertet": int(last["n"]),
        "angefragt": len(tickers),
        "ueberkauft_pct": round(float(last["ob"]), 1),
        "ueberverkauft_pct": round(float(last["os"]), 1),
        "verlauf": [{"date": d.strftime("%Y-%m-%d"), "ob": round(float(r.ob), 1), "os": round(float(r.os), 1), "n": int(r.n)}
                    for d, r in res.iterrows()],
    }
    with open("data/obos_sp500.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("ok", out["stand_kurse"], out["ueberkauft_pct"], out["ueberverkauft_pct"], out["aktien_bewertet"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # Fehler sichtbar machen, ohne eine alte gute Datei zu überschreiben
        print("FEHLER:", e, file=sys.stderr)
        sys.exit(1)
