#!/usr/bin/env python3
"""
fetch_carbon_traces.py — scarica tracce orarie di carbon intensity (life-cycle)
dall'API Electricity Maps (endpoint /v3/carbon-intensity/past-range) e le salva
come CSV per zona, NEL FORMATO letto dal provider `trace` di
src/metrics/carbon_client.py.

Perche': il data portal non sempre espone il download CSV; con un token che
abbia accesso a `past-range` si ottiene lo stesso dato (orario, life-cycle) via
API. Si fetcha UNA volta, si committano i CSV, e i run restano riproducibili.

Sicurezza: il token NON va nel codice ne' nel repo. Si legge da variabile
d'ambiente ELECTRICITYMAPS_TOKEN.

Uso:
    # PowerShell:   $env:ELECTRICITYMAPS_TOKEN = "..."
    # Git Bash:     export ELECTRICITYMAPS_TOKEN=...
    python scripts/fetch_carbon_traces.py \
        --zones DE FR PL IT-NO NO-NO1 ES --year 2025

Output: data/carbon_traces/<ZONE>_<YEAR>_hourly.csv
    Colonne: "Datetime (UTC)", "Carbon Intensity gCO2eq/kWh (Life cycle)"
    (il loader del provider `trace` rileva queste colonne automaticamente).

NB: il prefisso del file (= <ZONE>) deve combaciare col `region_code` del
cluster in config/local.yaml. Es. zona "NO-NO1" -> region_code "NO-NO1".
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

API = "https://api.electricitymaps.com/v3/carbon-intensity/past-range"


def _chunk_ranges(year: int, chunk_days: int = 10):
    """Genera (start, end) a finestre di <chunk_days> giorni sull'intero anno.

    L'API past-range limita i dati orari a 10 giorni per richiesta, quindi si
    spezza l'anno in finestre consecutive (end esclusivo → niente buchi/overlap;
    eventuali duplicati sono comunque deduplicati per timestamp a valle).
    """
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    step = timedelta(days=chunk_days)
    while start < year_end:
        end = min(start + step, year_end)
        yield start, end
        start = end


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def fetch_zone(zone: str, year: int, token: str, session: requests.Session,
               extra_params: dict, chunk_days: int = 10) -> list:
    """Ritorna [(datetime_str, ci_lifecycle), ...] ordinato, deduplicato per ora."""
    rows: dict = {}
    for start, end in _chunk_ranges(year, chunk_days):
        # Default: solo zone/start/end (come la curl del portale; past-range
        # restituisce gia' hourly + lifecycle di default). Param extra opzionali.
        params = {"zone": zone, "start": _iso(start), "end": _iso(end)}
        params.update(extra_params)
        data = None
        for attempt in range(4):
            r = session.get(API, headers={"auth-token": token},
                            params=params, timeout=30)
            if r.status_code == 429:           # rate limit: backoff e riprova
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 401:
                sys.exit("  [401] token non valido o senza accesso a past-range.")
            if not r.ok:
                body = (r.text or "")[:400]
                sys.exit(f"  [{r.status_code}] {zone} {start:%Y-%m-%d}: {body}")
            data = r.json().get("data", [])
            break
        if data is None:
            sys.exit(f"  [{zone}] rate-limit persistente su {start:%Y-%m-%d}.")
        for it in data:
            ci = it.get("carbonIntensity")
            dt = it.get("datetime")
            if ci is None or dt is None:
                continue
            rows[dt] = ci
        time.sleep(0.2)                        # cortesia verso l'API
    print(f"  {zone}: {len(rows)} ore totali nel {year}")
    return sorted(rows.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zones", nargs="+", required=True,
                    help="Zone EM (devono combaciare coi region_code in local.yaml)")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out", default=None, help="Dir di output (default: data/carbon_traces)")
    # Param extra OPZIONALI: di default NON inviati (la curl del portale non li
    # usa e past-range e' gia' hourly+lifecycle). Aggiungili solo se servono.
    ap.add_argument("--granularity", default=None,
                    help="temporalGranularity (es. hourly). Default: non inviato.")
    ap.add_argument("--emission-factor", default=None,
                    help="emissionFactorType (lifecycle|direct). Default: non inviato.")
    ap.add_argument("--chunk-days", type=int, default=9,
                    help="Ampiezza finestra per richiesta (limite API: 10 giorni; default 9 per sicurezza).")
    args = ap.parse_args()

    extra_params: dict = {}
    if args.granularity:
        extra_params["temporalGranularity"] = args.granularity
    if args.emission_factor:
        extra_params["emissionFactorType"] = args.emission_factor

    token = os.environ.get("ELECTRICITYMAPS_TOKEN")
    if not token:
        sys.exit("Manca ELECTRICITYMAPS_TOKEN. Esportalo (NON committare il token).")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = args.out or os.path.join(repo_root, "data", "carbon_traces")
    os.makedirs(out_dir, exist_ok=True)

    session = requests.Session()
    written = 0
    for zone in args.zones:
        print(f"[{zone}] fetching {args.year} ...")
        series = fetch_zone(zone, args.year, token, session, extra_params,
                            chunk_days=args.chunk_days)
        if not series:
            print(f"  [warn] {zone}: nessun dato — salto.")
            continue
        path = os.path.join(out_dir, f"{zone}_{args.year}_hourly.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Datetime (UTC)", "Carbon Intensity gCO2eq/kWh (Life cycle)"])
            for dt, ci in series:
                # "2025-01-01T00:00:00.000Z" -> "2025-01-01 00:00:00"
                d = dt.replace("Z", "").replace("T", " ").split(".")[0]
                w.writerow([d, ci])
        print(f"  -> {path}  ({len(series)} ore)")
        written += 1

    print(f"\nFatto: {written}/{len(args.zones)} zone scritte in {out_dir}")
    if written:
        print("Ora in config/carbon.yaml metti  provider: \"trace\"  e verifica i log "
              "'Trace CI <zona>: ... gCO2/kWh'.")


if __name__ == "__main__":
    main()
