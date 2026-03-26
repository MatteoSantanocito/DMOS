"""
analyze_flows.py — Report e filtro interattivo dei flussi Hubble

Uso:
  1. Cattura flussi durante un test:
       hubble observe --follow --output json --namespace online-boutique > results/flows/flows.json

  2. Analizza:
       python experiments/analyze_flows.py results/flows/flows.json

  3. Filtra:
       python experiments/analyze_flows.py results/flows/flows.json --src-cluster cluster1
       python experiments/analyze_flows.py results/flows/flows.json --dst-pod productcatalogservice
       python experiments/analyze_flows.py results/flows/flows.json --cross-cluster
       python experiments/analyze_flows.py results/flows/flows.json --flow 42
       python experiments/analyze_flows.py results/flows/flows.json --src-cluster cluster1 --dst-cluster cluster2
"""

import json
import sys
import argparse
from datetime import datetime
from collections import defaultdict


# ── Parser flussi Hubble ─────────────────────────────────────────────────────

def parse_flow(raw: dict) -> dict | None:
    """
    Normalizza un flow Hubble in un dict semplice.
    Ritorna None se il flow non è rilevante (es. health-check interno).
    """
    flow = raw.get("flow") or raw  # hubble observe --output json wraps in {"flow": ...}

    time_str = flow.get("time", "")
    try:
        ts = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except Exception:
        ts = None

    src  = flow.get("source", {}) or {}
    dst  = flow.get("destination", {}) or {}
    l7   = flow.get("l7", {}) or {}
    http = l7.get("http", {}) or {}

    src_cluster = src.get("cluster_name", "") or src.get("cluster", "") or _infer_cluster(src)
    dst_cluster = dst.get("cluster_name", "") or dst.get("cluster", "") or _infer_cluster(dst)

    src_pod  = src.get("pod_name", "")  or src.get("workloads", [{}])[0].get("name", "") if src.get("workloads") else src.get("pod_name", "")
    dst_pod  = dst.get("pod_name", "")  or dst.get("workloads", [{}])[0].get("name", "") if dst.get("workloads") else dst.get("pod_name", "")
    src_svc  = _service_name(src_pod)
    dst_svc  = _service_name(dst_pod)
    src_ns   = src.get("namespace", "")
    dst_ns   = dst.get("namespace", "")

    verdict  = flow.get("verdict", "")
    method   = http.get("method", "")
    url      = http.get("url", "")
    status   = http.get("code", 0)
    latency  = None
    if "response_time" in l7:
        latency = l7["response_time"]  # ns
    elif http.get("headers"):
        pass  # latency non sempre disponibile

    return {
        "ts":          ts,
        "time_str":    time_str[:23] if time_str else "",
        "src_cluster": src_cluster,
        "src_ns":      src_ns,
        "src_pod":     src_pod,
        "src_svc":     src_svc,
        "dst_cluster": dst_cluster,
        "dst_ns":      dst_ns,
        "dst_pod":     dst_pod,
        "dst_svc":     dst_svc,
        "verdict":     verdict,
        "method":      method,
        "url":         url,
        "status":      status,
        "latency_ms":  round(latency / 1e6, 2) if latency else None,
        "cross_cluster": src_cluster != dst_cluster and bool(src_cluster) and bool(dst_cluster),
        "_raw":        flow,
    }


def _infer_cluster(endpoint: dict) -> str:
    """Prova a inferire il cluster dall'IP o dal labels."""
    labels = endpoint.get("labels", [])
    for l in labels:
        if "cluster" in str(l).lower():
            return str(l).split("=")[-1].strip()
    return ""


def _service_name(pod_name: str) -> str:
    """Estrae il nome del servizio dal nome del pod (rimuove hash suffix)."""
    if not pod_name:
        return ""
    parts = pod_name.split("-")
    # I pod k8s hanno tipicamente: <deploy>-<replicaset-hash>-<pod-hash>
    # Rimuoviamo gli ultimi 2 segmenti se sembrano hash (5 chars alfanumeric)
    while len(parts) > 1 and len(parts[-1]) <= 5 and parts[-1].isalnum():
        parts.pop()
    return "-".join(parts)


# ── Caricamento file ─────────────────────────────────────────────────────────

def load_flows(path: str) -> list[dict]:
    flows = []
    errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                parsed = parse_flow(raw)
                if parsed:
                    parsed["_idx"] = i
                    flows.append(parsed)
            except json.JSONDecodeError:
                errors += 1
    if errors:
        print(f"  ⚠️  {errors} righe non parsabili (ignorate)")
    return flows


# ── Filtri ───────────────────────────────────────────────────────────────────

def apply_filters(flows: list[dict], args) -> list[dict]:
    result = flows

    if args.src_cluster:
        result = [f for f in result if args.src_cluster.lower() in f["src_cluster"].lower()]
    if args.dst_cluster:
        result = [f for f in result if args.dst_cluster.lower() in f["dst_cluster"].lower()]
    if args.src_pod:
        result = [f for f in result if args.src_pod.lower() in f["src_pod"].lower()]
    if args.dst_pod:
        result = [f for f in result if args.dst_pod.lower() in f["dst_pod"].lower()]
    if args.src_svc:
        result = [f for f in result if args.src_svc.lower() in f["src_svc"].lower()]
    if args.dst_svc:
        result = [f for f in result if args.dst_svc.lower() in f["dst_svc"].lower()]
    if args.cross_cluster:
        result = [f for f in result if f["cross_cluster"]]
    if args.verdict:
        result = [f for f in result if args.verdict.upper() in f["verdict"].upper()]
    if args.status:
        result = [f for f in result if str(f["status"]) == str(args.status)]
    if args.namespace:
        result = [f for f in result
                  if args.namespace.lower() in f["src_ns"].lower()
                  or args.namespace.lower() in f["dst_ns"].lower()]

    return result


# ── Report ───────────────────────────────────────────────────────────────────

def print_summary(flows: list[dict]):
    print(f"\n{'═'*70}")
    print(f"  HUBBLE FLOWS REPORT  —  {len(flows)} flussi")
    print(f"{'═'*70}")

    if not flows:
        print("  Nessun flusso trovato con i filtri applicati.")
        return

    # Intervallo temporale
    ts_list = [f["ts"] for f in flows if f["ts"]]
    if ts_list:
        print(f"  Periodo:  {min(ts_list).strftime('%H:%M:%S')} → {max(ts_list).strftime('%H:%M:%S')}")

    # Cross-cluster stats
    cross = [f for f in flows if f["cross_cluster"]]
    local = [f for f in flows if not f["cross_cluster"]]
    print(f"  Flussi locali:        {len(local):>6} ({100*len(local)/len(flows):.1f}%)")
    print(f"  Flussi cross-cluster: {len(cross):>6} ({100*len(cross)/len(flows):.1f}%)")

    # Verdict breakdown
    verdicts = defaultdict(int)
    for f in flows:
        verdicts[f["verdict"] or "UNKNOWN"] += 1
    print(f"\n  Verdict:")
    for v, cnt in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"    {v:<20} {cnt:>6}")

    # Top route: src_cluster/src_svc → dst_cluster/dst_svc
    routes = defaultdict(int)
    for f in flows:
        key = f"{f['src_cluster'] or '?'}/{f['src_svc'] or '?'} → {f['dst_cluster'] or '?'}/{f['dst_svc'] or '?'}"
        routes[key] += 1
    print(f"\n  Top 15 route (src_cluster/svc → dst_cluster/svc):")
    for route, cnt in sorted(routes.items(), key=lambda x: -x[1])[:15]:
        cross_tag = " ★" if route.split(" → ")[0].split("/")[0] != route.split(" → ")[1].split("/")[0] else ""
        print(f"    {cnt:>6}  {route}{cross_tag}")

    # Latenza (se disponibile)
    lat = [f["latency_ms"] for f in flows if f["latency_ms"] is not None]
    if lat:
        lat_sorted = sorted(lat)
        n = len(lat_sorted)
        print(f"\n  Latenza HTTP (dove disponibile, {n} flussi):")
        print(f"    avg={sum(lat)/n:.2f}ms  "
              f"p50={lat_sorted[int(n*0.50)]:.2f}ms  "
              f"p95={lat_sorted[int(n*0.95)]:.2f}ms  "
              f"p99={lat_sorted[int(n*0.99)]:.2f}ms  "
              f"max={max(lat):.2f}ms")

    # Status code breakdown
    statuses = defaultdict(int)
    for f in flows:
        if f["status"]:
            statuses[str(f["status"])] += 1
    if statuses:
        print(f"\n  HTTP Status codes:")
        for s, cnt in sorted(statuses.items()):
            print(f"    {s}: {cnt}")


def print_table(flows: list[dict], limit: int = 50):
    """Stampa tabella compatta dei flussi."""
    print(f"\n  {'#':>5}  {'Time':>8}  {'Src Cluster':>12}  {'Src Pod/Svc':>28}  "
          f"{'→':>2}  {'Dst Cluster':>12}  {'Dst Pod/Svc':>28}  {'St':>4}  {'Lat':>8}  Verdict")
    print(f"  {'-'*5}  {'-'*8}  {'-'*12}  {'-'*28}  {'-'*2}  {'-'*12}  {'-'*28}  {'-'*4}  {'-'*8}  {'-'*10}")

    shown = flows[:limit]
    for i, f in enumerate(shown):
        t   = f["time_str"][11:19] if f["time_str"] else ""
        sc  = (f["src_cluster"] or "?")[-12:]
        sp  = (f["src_pod"] or f["src_svc"] or "?")[-28:]
        dc  = (f["dst_cluster"] or "?")[-12:]
        dp  = (f["dst_pod"] or f["dst_svc"] or "?")[-28:]
        st  = str(f["status"]) if f["status"] else "-"
        lat = f"{f['latency_ms']:.1f}ms" if f["latency_ms"] else "-"
        vrd = f["verdict"][:10] if f["verdict"] else "-"
        cross_tag = "★" if f["cross_cluster"] else " "
        print(f"  {i:>5}  {t:>8}  {sc:>12}  {sp:>28} {cross_tag}→  {dc:>12}  {dp:>28}  {st:>4}  {lat:>8}  {vrd}")

    if len(flows) > limit:
        print(f"\n  ... e altri {len(flows) - limit} flussi. Usa --limit N per vederne di più.")


def print_flow_detail(flows: list[dict], idx: int):
    """Stampa il dettaglio completo di un singolo flusso."""
    if idx >= len(flows):
        print(f"  ❌ Indice {idx} fuori range (0–{len(flows)-1})")
        return

    f = flows[idx]
    print(f"\n{'═'*60}")
    print(f"  FLOW #{idx} — dettaglio")
    print(f"{'═'*60}")
    print(f"  Timestamp:       {f['time_str']}")
    print(f"  Verdict:         {f['verdict']}")
    print(f"  Cross-cluster:   {'Sì ★' if f['cross_cluster'] else 'No'}")
    print(f"\n  SOURCE")
    print(f"    Cluster:       {f['src_cluster'] or '(sconosciuto)'}")
    print(f"    Namespace:     {f['src_ns']}")
    print(f"    Pod:           {f['src_pod']}")
    print(f"    Servizio:      {f['src_svc']}")
    print(f"\n  DESTINATION")
    print(f"    Cluster:       {f['dst_cluster'] or '(sconosciuto)'}")
    print(f"    Namespace:     {f['dst_ns']}")
    print(f"    Pod:           {f['dst_pod']}")
    print(f"    Servizio:      {f['dst_svc']}")
    if f["method"] or f["url"]:
        print(f"\n  HTTP")
        print(f"    Method:        {f['method']}")
        print(f"    URL:           {f['url']}")
        print(f"    Status:        {f['status']}")
        print(f"    Latency:       {f['latency_ms']}ms" if f["latency_ms"] else "    Latency:       -")
    print(f"\n  RAW JSON:")
    print(json.dumps(f["_raw"], indent=4, default=str))


# ── Cross-cluster breakdown ───────────────────────────────────────────────────

def print_cross_cluster_breakdown(flows: list[dict]):
    """Breakdown dettagliato dei flussi cross-cluster."""
    cross = [f for f in flows if f["cross_cluster"]]
    if not cross:
        print("\n  Nessun flusso cross-cluster trovato.")
        return

    print(f"\n{'═'*70}")
    print(f"  CROSS-CLUSTER BREAKDOWN  —  {len(cross)} flussi")
    print(f"{'═'*70}")

    # Matrice cluster sorgente → cluster destinazione
    matrix = defaultdict(lambda: defaultdict(int))
    for f in cross:
        matrix[f["src_cluster"]][f["dst_cluster"]] += 1

    clusters = sorted(set(
        list(matrix.keys()) + [k for v in matrix.values() for k in v.keys()]
    ))

    print(f"\n  Matrice traffico cross-cluster (righe=src, colonne=dst):")
    header = f"  {'':>12}" + "".join(f"  {c:>12}" for c in clusters)
    print(header)
    for src in clusters:
        row = f"  {src:>12}"
        for dst in clusters:
            cnt = matrix[src][dst]
            row += f"  {cnt:>12}" if cnt else f"  {'—':>12}"
        print(row)

    # Top route cross-cluster con pod dettaglio
    routes_pods = defaultdict(int)
    for f in cross:
        key = (f["src_cluster"], f["src_svc"], f["dst_cluster"], f["dst_svc"])
        routes_pods[key] += 1

    print(f"\n  Route cross-cluster per servizio:")
    for (sc, ss, dc, ds), cnt in sorted(routes_pods.items(), key=lambda x: -x[1]):
        print(f"    {cnt:>5}  {sc}/{ss} → {dc}/{ds}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analizza e filtra flussi Hubble da file JSON"
    )
    parser.add_argument("file", help="File JSON prodotto da 'hubble observe --output json'")

    # Filtri
    parser.add_argument("--src-cluster",  help="Filtra per cluster sorgente (es. cluster1)")
    parser.add_argument("--dst-cluster",  help="Filtra per cluster destinazione")
    parser.add_argument("--src-pod",      help="Filtra per nome pod sorgente (match parziale)")
    parser.add_argument("--dst-pod",      help="Filtra per nome pod destinazione (match parziale)")
    parser.add_argument("--src-svc",      help="Filtra per servizio sorgente (es. frontend)")
    parser.add_argument("--dst-svc",      help="Filtra per servizio destinazione (es. productcatalogservice)")
    parser.add_argument("--namespace",    help="Filtra per namespace (src o dst)")
    parser.add_argument("--cross-cluster", action="store_true",
                        help="Mostra solo flussi cross-cluster")
    parser.add_argument("--verdict",      help="Filtra per verdict (es. FORWARDED, DROPPED)")
    parser.add_argument("--status",       help="Filtra per HTTP status code (es. 200, 500)")

    # Output
    parser.add_argument("--flow",   type=int, metavar="N",
                        help="Mostra dettaglio completo del flusso N (dopo i filtri)")
    parser.add_argument("--limit",  type=int, default=50,
                        help="Numero max di righe nella tabella (default: 50)")
    parser.add_argument("--no-table", action="store_true",
                        help="Mostra solo summary, senza tabella")
    parser.add_argument("--cross-breakdown", action="store_true",
                        help="Mostra breakdown dettagliato cross-cluster")

    args = parser.parse_args()

    print(f"\n  Caricamento {args.file}...")
    flows = load_flows(args.file)
    print(f"  {len(flows)} flussi caricati.")

    # Applica filtri
    filtered = apply_filters(flows, args)

    # Output
    if args.flow is not None:
        # Dettaglio singolo flusso
        print_flow_detail(filtered, args.flow)
    else:
        print_summary(filtered)
        if args.cross_breakdown:
            print_cross_cluster_breakdown(filtered)
        if not args.no_table:
            print_table(filtered, limit=args.limit)


if __name__ == "__main__":
    main()
