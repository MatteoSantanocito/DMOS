#!/usr/bin/env python3
"""journal_tests.py — the pre-specified tests of the journal campaign.

  python experiments/journal_tests.py --tsv experiments/results/_campaign_2026-11-02-definitive.tsv
  python experiments/journal_tests.py --proxy        # pilot runs, pipeline check only

The tests are the ones written in the header of
experiments/plans/2026-11-02-definitive.plan, before the campaign started:
exact permutation test on the difference of means (all C(nA+nB, nA)
relabellings, 252 for 5 vs 5), one-sided in the stated direction unless marked
two-sided, Holm within each family, alpha 0.05. Effects are the % change of
mean(A) over mean(B) with a 95% percentile bootstrap CI (10k resamples), plus
Cliff's delta. Only status OK rows of the campaign TSV are used.

Output (in --out, default paper/figures/journal): tests.md, tests.csv,
table_tests.tex. Metrics come from journal_data (same numbers as the figures)
and share its cache (--out/_cache).
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import journal_data as jd  # noqa: E402
from journal_figures import ARMS, REGIMES  # noqa: E402

ALPHA = 0.05


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(B > A) - P(B < A) over all pairs, in [-1, 1]."""
    gt = sum((bi > aj) for bi in b for aj in a)
    lt = sum((bi < aj) for bi in b for aj in a)
    return (gt - lt) / (len(a) * len(b))


def holm(pvalues: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj
METRICS = {   # key in journal_data.metrics -> (label, unit)
    "p95": ("p95 latency", "ms"),
    "fe_rm": ("Frontend replica-minutes", "replica-min"),
    "co2_ug_per_req": ("CO2 per request", "ug"),
    "ci_served_g_kwh": ("Carbon intensity served", "g/kWh"),
}
FAMILIES = {
    "F1": "Primary: DMOS vs HPA and vs even split",
    "F2": "Sustainability: carbon-aware weights and scale to zero",
    "F3": "Ablation: demand signal of the placement score",
}


@dataclass(frozen=True)
class Test:
    family: str
    regime: str
    a: str          # arm expected lower (or first arm of a two-sided test)
    b: str
    metric: str
    two_sided: bool = False

    @property
    def id(self) -> str:
        return f"{self.family}:{self.regime}:{self.a}<{self.b}:{self.metric}"


def preregistered() -> List[Test]:
    """Exactly the tests of the plan header (F1 9, F2 6, F3 3)."""
    t = []
    for reg in ("bal", "con", "con-tosg"):
        t += [Test("F1", reg, "reach-pool-ps", "hpa-none", "p95"),
              Test("F1", reg, "reach-pool-ps", "hpa-none", "fe_rm"),
              Test("F1", reg, "reach-pool-ps", "l2-pool-ps", "p95")]
    for reg in ("bal", "con"):
        t += [Test("F2", reg, "reach_green-pool-ps-min0", "reach-pool-ps-min0", "ci_served_g_kwh"),
              Test("F2", reg, "reach_green-pool-ps-min0", "reach-pool-ps-min0", "co2_ug_per_req"),
              Test("F2", reg, "reach-pool-ps-min0", "reach-pool-ps", "fe_rm")]
    t += [Test("F3", "con", "reach-pool-ps", "nca-pool-ps", "p95"),
          Test("F3", "con", "reach-pool-ps", "locality-ps", "p95"),
          Test("F3", "con", "nca-pool-ps", "locality-ps", "p95", two_sided=True)]
    return t


def perm_test(a, b, two_sided: bool = False) -> float:
    """Exact permutation p of mean(a) - mean(b); one-sided H1: mean(a) < mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pool, na = np.concatenate([a, b]), len(a)
    total = pool.sum()
    obs = a.mean() - b.mean()
    tol = 1e-9 * max(1.0, abs(obs))
    hits = n = 0
    for idx in combinations(range(len(pool)), na):
        sa = pool[list(idx)].sum()
        d = sa / na - (total - sa) / len(b)
        hits += (abs(d) >= abs(obs) - tol) if two_sided else (d <= obs + tol)
        n += 1
    return hits / n


def _values(ds: jd.Dataset, arm: str, regime: str, metric: str) -> np.ndarray:
    v = [ds.m(r).get(metric) for r in ds.group(arm, regime)]
    return np.asarray([x for x in v if x is not None and not math.isnan(x)], float)


def run_tests(ds: jd.Dataset, tests: List[Test]) -> pd.DataFrame:
    rows = []
    for t in tests:
        a, b = _values(ds, t.a, t.regime, t.metric), _values(ds, t.b, t.regime, t.metric)
        row = {"family": t.family, "regime": t.regime, "a": t.a, "b": t.b, "metric": t.metric,
               "sides": 2 if t.two_sided else 1, "n_a": len(a), "n_b": len(b),
               "mean_a": a.mean() if len(a) else np.nan, "mean_b": b.mean() if len(b) else np.nan,
               "sd_a": a.std(ddof=1) if len(a) > 1 else np.nan,
               "sd_b": b.std(ddof=1) if len(b) > 1 else np.nan, "p": np.nan}
        if len(a) >= 2 and len(b) >= 2:
            row["delta_pct"], row["ci_lo"], row["ci_hi"] = jd.boot_ratio_ci(a, b)
            row["cliff"] = -cliffs_delta(a, b)   # sign: < 0 when A tends to be lower
            row["p"] = perm_test(a, b, t.two_sided)
            row["p_min"] = 1 / math.comb(len(a) + len(b), len(a)) * (2 if t.two_sided else 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for fam in df.family.unique():   # Holm within each family, over the tests that ran
        idx = df.index[(df.family == fam) & df.p.notna()]
        if len(idx):
            df.loc[idx, "p_holm"] = holm(df.loc[idx, "p"].tolist())
    df["family_complete"] = df.family.map(df.p.notna().groupby(df.family).all())
    df["supported"] = (df.p_holm <= ALPHA) & df.family_complete
    return df


def config_ids(ds: jd.Dataset) -> Dict[str, List[str]]:
    """Distinct DMOS config_id per arm: each arm must have exactly one."""
    out: Dict[str, set] = {}
    for r in ds.runs:
        out.setdefault(r.arm, set()).add(r.config_id or "-")
    return {k: sorted(v) for k, v in out.items()}


# ── output ──────────────────────────────────────────────────────────────────

def _lbl(arm: str) -> str:
    return ARMS.get(arm, (arm,))[0]


def _fmt(x, d=1) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def _p(x) -> str:
    return "–" if x is None or math.isnan(x) else ("<0.001" if x < 0.001 else f"{x:.3f}")


def to_markdown(df: pd.DataFrame, cids: Dict[str, List[str]], proxy: bool) -> str:
    out = ["# Pre-specified tests (plan 2026-11-02-definitive)", ""]
    if proxy:
        out += ["**PROXY: pilot runs, not the campaign. Pipeline check only.**", ""]
    out += [f"Exact permutation test on the mean, one-sided unless marked; Holm within each family; "
            f"alpha = {ALPHA}. Δ = mean(A)/mean(B) − 1 with 95% bootstrap CI; Cliff's δ < 0 when A "
            "tends to be lower. A family whose tests did not all run is marked incomplete and "
            "no test in it counts as supported.", ""]
    for fam, title in FAMILIES.items():
        d = df[df.family == fam]
        if d.empty:
            continue
        done = bool(d.family_complete.iloc[0])
        out += [f"## {fam} — {title}" + ("" if done else " (INCOMPLETE)"), "",
                "| Regime | A | B | Metric | n | mean A | mean B | Δ % [95% CI] | Cliff δ | p | p Holm | |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            ok = "✓" if r.supported else ""
            out.append(f"| {REGIMES.get(r.regime, r.regime)} | {_lbl(r.a)} | {_lbl(r.b)} | "
                       f"{METRICS[r.metric][0]} ({METRICS[r.metric][1]}) | "
                       f"{r.n_a}/{r.n_b} | {_fmt(r.mean_a)} | {_fmt(r.mean_b)} | "
                       f"{_fmt(r.get('delta_pct'))} [{_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))}] | "
                       f"{_fmt(r.get('cliff'), 2)} | {_p(r.p)}"
                       f"{' (2-sided)' if r.sides == 2 else ''} | {_p(r.p_holm)} | {ok} |")
        out.append("")
    out += ["## DMOS configuration per arm", "",
            "Each DMOS arm must have one config_id (static arms have none: '-').", ""]
    for arm, ids in cids.items():
        out.append(f"- {_lbl(arm)} (`{arm}`): " + ", ".join(f"`{i}`" for i in ids)
                   + ("  **MORE THAN ONE**" if len(ids) > 1 else ""))
    return "\n".join(out) + "\n"


def to_latex(df: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{llllrrl}", r"\toprule",
             r"Family & Regime & Comparison & Metric & $n$ & $\Delta$ [95\% CI] (\%) & $p_{\mathrm{Holm}}$ \\",
             r"\midrule"]
    for fam in FAMILIES:
        d = df[df.family == fam]
        for k, (_, r) in enumerate(d.iterrows()):
            rel = r"\neq" if r.sides == 2 else "<"
            p = _p(r.p_holm).replace("<", "$<$")
            lines.append(f"{fam if k == 0 else ''} & {REGIMES.get(r.regime, r.regime)} & "
                         f"{_lbl(r.a)} ${rel}$ {_lbl(r.b)} & {METRICS[r.metric][0]} & {r.n_a}/{r.n_b} & "
                         f"{_fmt(r.get('delta_pct'))} [{_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))}] & "
                         f"{p}{'$^{*}$' if r.supported else ''} \\\\")
        if not d.empty and fam != list(FAMILIES)[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines).replace("CO2", r"CO$_2$").replace("%]", r"\%]").replace("→", r"$\to$") + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--tsv", type=Path, action="append", default=[])
    ap.add_argument("--proxy", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE.parent / "paper" / "figures" / "journal")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    if not a.proxy and not a.tsv:
        ap.error("give --tsv <campaign tsv> (or --proxy)")
    a.out.mkdir(parents=True, exist_ok=True)
    ds = jd.load(a.results, a.tsv, a.proxy, None if a.no_cache else a.out / "_cache")
    df = run_tests(ds, preregistered())
    cids = config_ids(ds)
    df.to_csv(a.out / "tests.csv", index=False)
    (a.out / "tests.md").write_text(to_markdown(df, cids, a.proxy), encoding="utf-8")
    (a.out / "table_tests.tex").write_text(to_latex(df), encoding="utf-8")
    for fam in FAMILIES:
        d = df[df.family == fam]
        print(f"{fam}: {int(d.supported.sum())}/{len(d)} supported"
              + ("" if d.family_complete.all() else " (family incomplete)"))
    bad = {k: v for k, v in cids.items() if len(v) > 1}
    if bad:
        print(f"[warn] arms with more than one config_id: {bad}", file=sys.stderr)
    print(f"-> {a.out / 'tests.md'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        assert math.isclose(perm_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]), 1 / 252)
        assert math.isclose(perm_test([6, 7, 8, 9, 10], [1, 2, 3, 4, 5]), 1.0)
        assert math.isclose(perm_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], two_sided=True), 2 / 252)
        assert math.isclose(perm_test([1, 1, 1], [1, 1, 1]), 1.0)              # ties count as extreme
        assert math.isclose(perm_test([1, 2], [3, 4, 5]), 1 / 10)              # unequal n: C(5,2)
        assert len(preregistered()) == 18
        assert [len([t for t in preregistered() if t.family == f]) for f in FAMILIES] == [9, 6, 3]
        print("journal_tests self-check OK")
        sys.exit(0)
    sys.exit(main())
