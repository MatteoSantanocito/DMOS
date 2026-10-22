"""
paper_lib.py — Shared library for DMOS paper-quality analysis pipeline
======================================================================

Modulo condiviso da `paper_plot_trajectories.py`, `paper_plot_supplementary.py`,
`paper_stat_compare.py`, `paper_report.py`. Espone:

- Costanti: SCENARIOS, SCENARIO_COLORS/LABELS, CLUSTER_COLORS, PHASE_DEFS
- Style: setup_matplotlib(target_dpi=600) → applica rcParams uniformi
- Dataclass: `Run` (un singolo run flash-crowd) + `ScenarioGroup` (N ripetizioni)
- Loader: `discover_runs(results_dir)` → {scenario: [Run, Run, Run]}
- Stat helpers: `bootstrap_ci(values, n_iter=1000)`, `welch_ttest(a, b)`, `cohen_d(a, b)`
- Binning: `bin_timeline(df, bin_sec=5)` → DataFrame con mean per bin
- Phase: `assign_phase(elapsed_s)` → 'baseline'|'spike'|'peak'|'decay'|'cooldown'
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Console UTF-8 (fix Windows cp1252) ──────────────────────────────────────
# Su Windows la console di default e' cp1252: i caratteri box-drawing / frecce
# usati nei log di questi script causano UnicodeEncodeError. Forziamo UTF-8 su
# stdout/stderr di ogni processo che importa paper_lib (cioe' tutti e 5 gli
# script paper_*.py), cosi' non serve esportare PYTHONUTF8 a mano.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        if (_stream is not None and _stream.encoding
                and _stream.encoding.lower() not in ('utf-8', 'utf8')):
            _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ── Style / Palette ────────────────────────────────────────────────────────

TARGET_DPI = 600    # paper-quality
PREVIEW_DPI = 150

# Le 3 strategie dello studio, allineate ai modi realmente eseguibili dal
# dmos_main (l'unico flag disponibile e' --no-scheduling).
# Ordinate dal peggio al meglio: visualizzazioni gradate da sinistra a destra.
#   dmos-off   : DMOS non gira — allocazione statica, nessuna orchestrazione
#   dmos-l2    : solo Level 2 — `python -m src.dmos_main --no-scheduling`
#                (scaling PD-controller, distribuzione uniforme N/K)
#   dmos-l1l2  : DMOS completo — `python -m src.dmos_main`
#                (Level 1 geo-aware scheduling + Level 2 scaling)
SCENARIOS: List[str] = ['dmos-off', 'dmos-l2', 'dmos-l1l2']

SCENARIO_LABELS: Dict[str, str] = {
    'dmos-off':   'DMOS off (static)',
    'dmos-l2':    'DMOS L2 (scaling only)',
    'dmos-l1l2':  'DMOS L1+L2 (full)',
}

# Palette consistente: rosso=worst, blu scuro=best (tesi convention)
SCENARIO_COLORS: Dict[str, str] = {
    'dmos-off':   '#C62828',   # red    (worst)
    'dmos-l2':    '#F57C00',   # orange (mid)
    'dmos-l1l2':  '#1565C0',   # blue   (best)
}

# Hatches per bar charts in B&W print
SCENARIO_HATCHES: Dict[str, str] = {
    'dmos-off':   '\\\\',
    'dmos-l2':    '//',
    'dmos-l1l2':  '',
}

# ── Arm / regime taxonomy ───────────────────────────────────────────────────
#
# Nome della cartella di un run:  <ts>_dmos-<arm>__<regime>
# (doppio underscore fra arm e regime; l'arm puo' contenere '-' e '_' singoli).
# Il significato di OGNI nome (che configurazione DMOS, che carico k6) e'
# definito in UN solo punto, experiments/run_naming.py, usato anche dagli
# orchestratori:  python experiments/run_naming.py list
#   arm     off | l2 | l1l2 | <modello di score> (nca, gate_g0, ...) |
#           profile-<profilo> | legacy-default | legacy-gate1 | legacy-nofloor
#   regime  <domanda>[-<cluster caldo>][-p<picco>]   es. con, bal-sg, con-p400
# I run "legacy" senza '__' restano validi: arm=label intera, regime=None.
# Le label delle campagne passate restano leggibili: dmos-latonly/netonly/
# carbon (oggi profile-*), dmos-l1l2coh (oggi l1l2: e' la configurazione della
# campagna NCA 2026), regime con400 (oggi con-p400).

import sys as _sys_path_fix  # noqa: E402
_EXPERIMENTS_DIR = str(Path(__file__).resolve().parent)
if _EXPERIMENTS_DIR not in _sys_path_fix.path:
    _sys_path_fix.path.insert(0, _EXPERIMENTS_DIR)
import run_naming as _naming  # noqa: E402

ARM_LABELS: Dict[str, str] = {
    # core (paper NCA 2026)
    'dmos-off':     'DMOS off (static)',
    'dmos-l2':      'DMOS L2 (scaling only)',
    'dmos-l1l2':    'DMOS L1+L2 (full)',
    # modelli di score (config/weights.yaml → score_models)
    'dmos-nca':             'DMOS (NCA model, g=2)',
    'dmos-gate_g0':         'Gate g=0 (weighted sum)',
    'dmos-gate_g1':         'Gate g=1',
    'dmos-gate_g3':         'Gate g=3',
    'dmos-additive_dem50':  'Demand as weight (0.5)',
    'dmos-locality':        'Demand on proximity only',
    'dmos-latency_first':   'Latency-first',
    'dmos-capacity_first':  'Capacity-first',
    'dmos-carbon_first':    'Carbon-first',
    # profili legacy (DMOS_SCORE_PROFILE) e arm legacy a switch d'ambiente
    'dmos-profile-latonly': 'Latency-only (LAIS-like)',
    'dmos-profile-netonly': 'Locality-only (NetMARKS-like)',
    'dmos-profile-carbon':  'Carbon-only (CASPER-like)',
    'dmos-legacy-gate1':    'Legacy gate g=1',
    'dmos-legacy-default':  'Legacy default (demand floor)',
    'dmos-legacy-nofloor':  'Legacy additive (no floor)',
    # label delle campagne passate
    'dmos-l1l2coh': 'DMOS L1+L2 (full)',
    'dmos-latonly': 'Latency-only (LAIS-like)',
    'dmos-netonly': 'Locality-only (NetMARKS-like)',
    'dmos-carbon':  'Carbon-only (CASPER-like)',
}

ARM_COLORS: Dict[str, str] = {
    'dmos-off':     '#C62828',   # red    (worst)
    'dmos-l2':      '#F57C00',   # orange (mid)
    'dmos-l1l2':    '#1565C0',   # blue   (best / full DMOS)
    'dmos-nca':             '#0D47A1',   # dark blue (paper model)
    'dmos-gate_g0':         '#9E9E9E',   # grey  (no gate)
    'dmos-gate_g1':         '#64B5F6',   # light blue
    'dmos-gate_g3':         '#1A237E',   # indigo
    'dmos-additive_dem50':  '#AD1457',   # magenta
    'dmos-locality':        '#6A1B9A',   # purple
    'dmos-latency_first':   '#2E7D32',   # green
    'dmos-capacity_first':  '#EF6C00',   # dark orange
    'dmos-carbon_first':    '#00838F',   # teal
    'dmos-profile-latonly': '#2E7D32',
    'dmos-profile-netonly': '#6A1B9A',
    'dmos-profile-carbon':  '#00838F',
    'dmos-legacy-gate1':    '#5C6BC0',
    'dmos-legacy-default':  '#90A4AE',
    'dmos-legacy-nofloor':  '#78909C',
    'dmos-l1l2coh': '#1565C0',
    'dmos-latonly': '#2E7D32',   # green
    'dmos-netonly': '#6A1B9A',   # purple
    'dmos-carbon':  '#00838F',   # teal
}

ARM_HATCHES: Dict[str, str] = {
    'dmos-off':     '\\\\',
    'dmos-l2':      '//',
    'dmos-l1l2':    '',
    'dmos-nca':     '',
    'dmos-gate_g0': '--',
    'dmos-additive_dem50': '++',
    'dmos-latency_first':  'xx',
    'dmos-capacity_first': '||',
    'dmos-carbon_first':   'oo',
    'dmos-profile-latonly': 'xx',
    'dmos-profile-netonly': '..',
    'dmos-profile-carbon':  'oo',
    'dmos-l1l2coh': '',
    'dmos-latonly': 'xx',
    'dmos-netonly': '..',
    'dmos-carbon':  'oo',
}

# Regimi di domanda (demand skew). `REGIME_SKEW` e' la coordinata X monotona
# crescente per il plot "p95 vs skew": la FRAZIONE DI DOMANDA sul cluster piu'
# caldo (hottest-cluster share) durante la fase di picco (0.167 = uniforme su
# 6 cluster; 1.0 = tutta la domanda su un solo cluster). I vettori di pesi k6
# di ogni regime sono in run_naming.DEMAND_WEIGHTS. Le varianti di cluster
# caldo e di picco (con-sg, con-p400, ...) hanno la stessa quota del regime
# base: usare regime_skew()/regime_label(), che le riconoscono.
REGIME_LABELS: Dict[str, str] = {
    'uni':    'Uniform',
    'bal':    'Balanced',
    's50':    'Skew 50%',
    's65':    'Skew 65%',
    's75':    'Skew 75%',
    'con':    'Concentrated',
    'con400': 'Concentrated @400rps',
}

REGIME_ORDER: List[str] = ['uni', 'bal', 's50', 's65', 's75', 'con', 'con400']

REGIME_SKEW: Dict[str, float] = {
    'uni':    0.167,
    'bal':    0.40,
    's50':    0.50,
    's65':    0.65,
    's75':    0.75,
    'con':    0.85,
    'con400': 0.85,
}


def parse_arm_regime(scenario: str) -> Tuple[str, Optional[str]]:
    """
    Splitta un'etichetta scenario '<arm>__<regime>' in (arm, regime).
    Se non c'e' '__', regime=None e arm=scenario (run legacy a 3 arm).
    """
    if '__' in scenario:
        arm, regime = scenario.split('__', 1)
        return arm, regime
    return scenario, None


def regime_skew(regime: Optional[str], default: Optional[float] = None) -> Optional[float]:
    """Quota del cluster caldo di un regime, anche nelle varianti
    (con-sg, s50-p500, con400 ...). `default` se il regime non e' riconosciuto."""
    if regime is None:
        return default
    if regime in REGIME_SKEW:
        return REGIME_SKEW[regime]
    share = _naming.regime_share(regime)
    return default if share is None else share


def regime_label(regime: Optional[str]) -> str:
    """Etichetta leggibile di un regime, anche nelle varianti."""
    if regime is None:
        return ''
    if regime in REGIME_LABELS:
        return REGIME_LABELS[regime]
    return _naming.regime_label(regime)


def scenario_label(scenario: str) -> str:
    """Label leggibile per uno scenario, composito o no."""
    if scenario in SCENARIO_LABELS:
        return SCENARIO_LABELS[scenario]
    arm, regime = parse_arm_regime(scenario)
    base = ARM_LABELS.get(arm, SCENARIO_LABELS.get(arm, arm))
    if regime:
        return f"{base} — {regime_label(regime)}"
    return base


def scenario_color(scenario: str) -> str:
    """Colore per uno scenario, composito o no (deciso dall'arm)."""
    if scenario in SCENARIO_COLORS:
        return SCENARIO_COLORS[scenario]
    arm, _ = parse_arm_regime(scenario)
    return ARM_COLORS.get(arm, SCENARIO_COLORS.get(arm, '#888888'))


def scenario_hatch(scenario: str) -> str:
    """Hatch per uno scenario, composito o no (deciso dall'arm)."""
    if scenario in SCENARIO_HATCHES:
        return SCENARIO_HATCHES[scenario]
    arm, _ = parse_arm_regime(scenario)
    return ARM_HATCHES.get(arm, SCENARIO_HATCHES.get(arm, ''))

# Palette per i 6 cluster, ispirata alla bandiera/regione di appartenenza
CLUSTER_COLORS: Dict[str, str] = {
    'ms-cluster-01': '#2E7D32',   # green   (DE)
    'ms-cluster-02': '#1565C0',   # blue    (FR)
    'ms-cluster-03': '#6A1B9A',   # purple  (PL)
    'ms-cluster-04': '#C62828',   # red     (US-VA)
    'ms-cluster-05': '#F57C00',   # orange  (SG)
    'ms-cluster-06': '#0288D1',   # cyan    (NO)
}

CLUSTER_REGIONS: Dict[str, str] = {
    'ms-cluster-01': 'EU-DE (Frankfurt)',
    'ms-cluster-02': 'EU-FR (Paris)',
    'ms-cluster-03': 'EU-PL (Warsaw)',
    'ms-cluster-04': 'US-VA (Virginia)',
    'ms-cluster-05': 'APAC-SG (Singapore)',
    'ms-cluster-06': 'EU-NO (Oslo)',
}

# 5 phases del test, in secondi (allineato a k6_flash_crowd.js)
# (t_start, t_end, name, light_color_for_shading)
PHASE_DEFS: List[Tuple[int, int, str, str]] = [
    (  0, 180, 'baseline', '#E3F2FD'),   # light blue
    (180, 300, 'spike',    '#FFF9C4'),   # light yellow
    (300, 600, 'peak',     '#FFCDD2'),   # light red
    (600, 720, 'decay',    '#FFF9C4'),   # light yellow
    (720, 900, 'cooldown', '#E8F5E9'),   # light green
]
PHASES = [p[2] for p in PHASE_DEFS]

BIN_SEC_DEFAULT = 5    # bin per timeline plots

# Gate di validita': un run e' escluso dall'analisi se k6 ha droppato
# >= 0.5% delle iterazioni offerte (VU starvation → carico non consegnato
# → arm incomparabile). Si applica al dato grezzo `dropped_pct`, non al
# booleano `run_valid` congelato nel summary, cosi' tutti i run sono
# giudicati con lo stesso metro. Allineato a MAX_DROPPED_PCT in k6_flash_crowd.js.
MAX_DROPPED_PCT = 0.005


# ── Carbon model (per Fig 9 / H4) ───────────────────────────────────────────
#
# Modello di potenza per nodo: lineare in CPU utilization.
#     P(cpu%) = P_idle + (P_max - P_idle) * cpu% / 100
#
# I valori sono per tier hardware. Sono APPROSSIMAZIONI documentate:
# sufficienti per il confronto RELATIVO tra arm (che e' quello che conta
# per il paper), inadeguati per claim assoluti sulla potenza esatta.
#
# Fonti per i valori (ordini di grandezza):
#   - HIGH 8 vCPU: ~15W idle / ~100W max  (server-class node)
#   - MEDIUM 6 vCPU: ~12W idle / ~75W max
#   - LOW 4 vCPU: ~10W idle / ~50W max
#
# Le carbon intensities (gCO2/kWh) per regione sono in config/local.yaml
# (campo `carbon.baseline_gco2_kwh` per cluster).
TIER_POWER_MODEL = {
    'HIGH':   {'p_idle': 15.0, 'p_max': 100.0},
    'MEDIUM': {'p_idle': 12.0, 'p_max':  75.0},
    'LOW':    {'p_idle': 10.0, 'p_max':  50.0},
}


def load_cluster_carbon_config(local_yaml: Path = None) -> Dict[str, dict]:
    """
    Legge config/local.yaml e restituisce, per cluster:
        {name: {'gco2_per_kwh', 'cpu_cores', 'tier', 'region', 'p_idle', 'p_max'}}
    Il tier e' derivato dai vCPU (HIGH>=8, MEDIUM>=6, LOW altrimenti).
    """
    import yaml as _yaml
    if local_yaml is None:
        local_yaml = Path(__file__).resolve().parent.parent / 'config' / 'local.yaml'
    with open(local_yaml, encoding='utf-8') as f:
        d = _yaml.safe_load(f)
    out = {}
    for c in d.get('clusters', []):
        cores = c.get('resources', {}).get('cpu_cores', 4)
        tier = 'HIGH' if cores >= 8 else ('MEDIUM' if cores >= 6 else 'LOW')
        tp = TIER_POWER_MODEL[tier]
        out[c['name']] = {
            'gco2_per_kwh': c.get('carbon', {}).get('baseline_gco2_kwh', 400.0),
            'cpu_cores':    cores,
            'tier':         tier,
            'region':       c.get('carbon', {}).get('region_code',
                                                     c.get('region', '?')),
            'p_idle':       tp['p_idle'],
            'p_max':        tp['p_max'],
        }
    return out


def compute_carbon_per_run(run: 'Run',
                           cluster_carbon: Optional[Dict[str, dict]] = None
                           ) -> Optional[Dict]:
    """
    Calcola le emissioni cumulative di CO2 di un run da `dmos_live.jsonl`.

    Per ogni snapshot di ogni cluster:
        power_W   = p_idle + (p_max - p_idle) * node_cpu_pct / 100
        energy_J  = power_W * dt    (dt = differenza temporale fra snapshot)
        co2_g     = energy_J / 3.6e6 * gco2_per_kwh

    Restituisce:
        {
          'per_cluster': {cluster_name: {'co2_g', 'energy_kWh', 'gco2_per_kwh',
                                          'tier', 'region', 'avg_cpu_pct'}},
          'total_co2_g':  ...,
          'total_co2_kg': ...,
        }
    o None se i dati timeline mancano.
    """
    if run.timeline_df is None or run.timeline_df.empty:
        return None
    if cluster_carbon is None:
        cluster_carbon = load_cluster_carbon_config()

    df = run.timeline_df.copy()
    if 'cpu_rate' not in df.columns or 'elapsed_s' not in df.columns:
        return None

    per_cluster: Dict[str, dict] = {}
    for cluster_name, sub in df.groupby('cluster'):
        cfg = cluster_carbon.get(str(cluster_name))
        if cfg is None:
            continue
        sub = sub.sort_values('elapsed_s').copy()
        # interval tra snapshot (s): se manca, fallback a 5s (collector default)
        dt = sub['elapsed_s'].diff().fillna(5.0).clip(lower=0.0)
        cpu = sub['cpu_rate'].fillna(0.0).clip(lower=0.0, upper=100.0)
        # Potenza istantanea (W)
        power_W = cfg['p_idle'] + (cfg['p_max'] - cfg['p_idle']) * cpu / 100.0
        # Energia per intervallo (J = W*s) -> kWh
        energy_Wh = (power_W * dt) / 3600.0
        energy_kWh = energy_Wh / 1000.0
        # CO2 (g) = energia (kWh) * intensita' (gCO2/kWh)
        co2_g = energy_kWh * cfg['gco2_per_kwh']
        per_cluster[str(cluster_name)] = {
            'co2_g':         float(co2_g.sum()),
            'energy_kWh':    float(energy_kWh.sum()),
            'gco2_per_kwh':  float(cfg['gco2_per_kwh']),
            'tier':          cfg['tier'],
            'region':        cfg['region'],
            'avg_cpu_pct':   float(cpu.mean()),
        }

    total_g = sum(v['co2_g'] for v in per_cluster.values())
    return {
        'per_cluster':   per_cluster,
        'total_co2_g':   total_g,
        'total_co2_kg':  total_g / 1000.0,
    }


def carbon_per_arm(groups: Dict[str, 'ScenarioGroup']) -> Dict[str, np.ndarray]:
    """
    Per ogni arm restituisce un np.array con il CO2 totale (g) di ogni run.
    Utile per bootstrap CI e t-test sulla metrica carbon.
    """
    cfg = load_cluster_carbon_config()
    out: Dict[str, np.ndarray] = {}
    for scen, group in groups.items():
        vals = []
        for run in group.runs:
            res = compute_carbon_per_run(run, cluster_carbon=cfg)
            if res is not None:
                vals.append(res['total_co2_g'])
        if vals:
            out[scen] = np.array(vals, dtype=float)
    return out


def setup_matplotlib(target_dpi: int = TARGET_DPI, font_size: int = 10):
    """Setta rcParams uniformi per tutte le figure del paper."""
    import matplotlib
    matplotlib.use('Agg')   # backend non-interactive (safe in CI / headless)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'figure.dpi':       PREVIEW_DPI,
        'savefig.dpi':      target_dpi,
        'font.family':      'sans-serif',
        'font.sans-serif':  ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size':        font_size,
        'axes.titlesize':   font_size + 2,
        'axes.titleweight': 'bold',
        'axes.labelsize':   font_size,
        'axes.grid':        True,
        'grid.alpha':       0.3,
        'grid.linestyle':   '--',
        'legend.fontsize':  font_size - 1,
        'legend.framealpha': 0.9,
        'lines.linewidth':  1.5,
        'savefig.bbox':     'tight',
        'savefig.pad_inches': 0.1,
        'pdf.fonttype':     42,   # embed fonts, no Type 3 (LaTeX safe)
        'ps.fonttype':      42,
    })


# ── Dataclass: Run e ScenarioGroup ─────────────────────────────────────────

@dataclass
class Run:
    """Un singolo run di flash-crowd test, caricato da experiments/results/<ts>_<label>/"""
    run_dir: Path
    scenario: str               # es. 'dmos-l1l2' (estratto da nome dir)
    timestamp: str              # es. '20260518_143000' (estratto da nome dir)
    repetition: int = 1         # assegnato in post da discover_runs
    manifest: dict = field(default_factory=dict)
    k6_summary: dict = field(default_factory=dict)
    timeline_df: Optional[pd.DataFrame] = None
    prom_pre: List[dict] = field(default_factory=list)
    prom_post: List[dict] = field(default_factory=list)
    # Provenance (runs from the schema-2 harness; empty for legacy runs)
    dmos_config: dict = field(default_factory=dict)     # dmos_config.json
    env_state_pre: dict = field(default_factory=dict)   # env_state_pre.json
    env_state_post: dict = field(default_factory=dict)  # env_state_post.json

    @property
    def label(self) -> str:
        return scenario_label(self.scenario)

    @property
    def color(self) -> str:
        return scenario_color(self.scenario)

    @property
    def hatch(self) -> str:
        return scenario_hatch(self.scenario)

    @property
    def config_id(self) -> Optional[str]:
        """Fingerprint of the DMOS configuration that produced the run
        ('off' for --no-dmos arms, None for legacy runs without provenance)."""
        if self.dmos_config.get('config_id'):
            return self.dmos_config['config_id']
        if self.manifest.get('no_dmos'):
            return 'off'
        return None

    @property
    def provenance(self) -> Optional[dict]:
        """manifest['provenance'] (checks, ok) — None for legacy runs."""
        return self.manifest.get('provenance')

    def is_complete(self) -> bool:
        """True se il run ha tutti i file principali (manifest + k6_summary minimo)."""
        return bool(self.manifest) and bool(self.k6_summary)


@dataclass
class ScenarioGroup:
    """Gruppo di N ripetizioni dello stesso scenario."""
    scenario: str
    runs: List[Run] = field(default_factory=list)

    @property
    def label(self) -> str:
        return scenario_label(self.scenario)

    @property
    def color(self) -> str:
        return scenario_color(self.scenario)

    @property
    def n_repetitions(self) -> int:
        return len(self.runs)

    def metric_values_per_run(self, key_path: str) -> np.ndarray:
        """
        Estrae un valore scalare da `k6_summary` per ogni run del gruppo.
        `key_path` e' tipo 'results.per_phase.peak.p95' (dot-separated).
        """
        vals = []
        for r in self.runs:
            cur: object = r.k6_summary
            try:
                for k in key_path.split('.'):
                    if isinstance(cur, dict):
                        cur = cur.get(k)
                    else:
                        cur = None
                        break
                if cur is not None and not isinstance(cur, (dict, list)):
                    vals.append(float(cur))
            except (TypeError, ValueError):
                continue
        return np.array(vals)


# ── Loader ──────────────────────────────────────────────────────────────────

# Pattern naming directory: <YYYYMMDD>_<HHMMSS>_<scenario>
# es. 20260518_143000_dmos-l1l2
_RUN_DIR_RE = re.compile(r'^(\d{8})_(\d{6})_(.+)$')


def discover_runs(results_dir: Path,
                  scenario_filter: Optional[List[str]] = None,
                  load_timeline: bool = True,
                  provenance_gate: bool = True) -> Dict[str, ScenarioGroup]:
    """
    Scopre tutti i run nella `results_dir`, raggruppati per scenario.

    Args:
        results_dir: typically `Path('experiments/results')`
        scenario_filter: se non None, mantiene solo questi scenari
        load_timeline: se True, carica anche timeline.csv (puo' essere lento)
        provenance_gate: se True (default) scarta i run il cui manifest
            registra errori di provenienza (configurazione DMOS ignota o
            cambiata durante il run, netem assente prima/dopo, DMOS attivo
            in un arm --no-dmos, ...). I run legacy senza provenienza restano.

    Returns:
        {scenario_name: ScenarioGroup} ordinato per SCENARIOS list.
        Stampa un avviso se uno scenario mescola configurazioni DMOS diverse
        (stessa label, config_id diversi): quei run NON sono ripetizioni.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    groups: Dict[str, ScenarioGroup] = {}

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        m = _RUN_DIR_RE.match(subdir.name)
        if not m:
            continue
        date_part, time_part, scenario = m.group(1), m.group(2), m.group(3)
        timestamp = f"{date_part}_{time_part}"

        if scenario_filter and scenario not in scenario_filter:
            continue

        run = _load_single_run(subdir, scenario, timestamp, load_timeline=load_timeline)
        if not run.is_complete():
            print(f"  [warn] skipping incomplete run: {subdir.name}")
            continue

        # Scarta i run dove k6 non ha consegnato il rate offerto (VU
        # starvation): l'arm ha ricevuto meno carico ed e' incomparabile.
        # Si applica il gate CORRENTE al dato grezzo `dropped_pct`, non al
        # booleano `run_valid` congelato nel summary — cosi' tutti i run sono
        # giudicati con lo stesso metro a prescindere da quando sono girati.
        # I run vecchi senza il campo `dropped_pct` restano inclusi.
        _dropped_pct = run.k6_summary.get('validity', {}).get('dropped_pct')
        if _dropped_pct is not None and _dropped_pct >= MAX_DROPPED_PCT:
            print(f"  [warn] skipping run (k6 VU-starved, dropped "
                  f"{_dropped_pct*100:.2f}% >= {MAX_DROPPED_PCT*100:.1f}%): {subdir.name}")
            continue

        # Gate di provenienza: un run con errori di provenienza non e'
        # interpretabile (algoritmo ignoto o testbed non nello stato atteso).
        prov = run.provenance
        if provenance_gate and prov is not None and not prov.get('ok', True):
            codes = [c['code'] for c in prov.get('checks', []) if c.get('level') == 'error']
            print(f"  [warn] skipping run (provenance errors: {', '.join(codes)}): {subdir.name}")
            continue

        groups.setdefault(scenario, ScenarioGroup(scenario=scenario)).runs.append(run)

    # Assegna numero di ripetizione in base a ordine cronologico
    for group in groups.values():
        group.runs.sort(key=lambda r: r.timestamp)
        for i, run in enumerate(group.runs, start=1):
            run.repetition = i
        # Ripetizioni = stessa configurazione. Se una label raccoglie run con
        # config_id diversi (es. "l1l2" lanciato con env diversi), avvisa.
        ids = {r.config_id for r in group.runs if r.config_id is not None}
        if len(ids) > 1:
            print(f"  [warn] scenario '{group.scenario}' MIXES {len(ids)} DMOS configurations "
                  f"{sorted(ids)}: these runs are NOT repetitions of the same algorithm "
                  f"(see: python experiments/run_provenance.py audit)")

    # Riordina dict in modo che SCENARIOS-ordered scenari vengano prima,
    # poi gli altri (custom) in fondo
    ordered: Dict[str, ScenarioGroup] = {}
    for s in SCENARIOS:
        if s in groups:
            ordered[s] = groups[s]
    for s in groups:
        if s not in ordered:
            ordered[s] = groups[s]
    return ordered


def _load_single_run(run_dir: Path, scenario: str, timestamp: str,
                     load_timeline: bool = True) -> Run:
    """Carica un singolo run dir nella struttura Run."""
    run = Run(run_dir=run_dir, scenario=scenario, timestamp=timestamp)

    # manifest.json (config + git commit)
    mp = run_dir / 'manifest.json'
    if mp.is_file():
        try:
            run.manifest = json.loads(mp.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            print(f"  [warn] manifest.json malformato in {run_dir.name}: {e}")

    # Summary k6 (output di handleSummary). Nome attuale prodotto da
    # run_flash_crowd.sh: flash_crowd_summary.json. Fallback al vecchio
    # k6_summary.json per retro-compatibilita' con run pre-2026.
    for sname in ('flash_crowd_summary.json', 'k6_summary.json'):
        sp = run_dir / sname
        if sp.is_file():
            try:
                run.k6_summary = json.loads(sp.read_text(encoding='utf-8'))
                break
            except json.JSONDecodeError as e:
                print(f"  [warn] {sname} malformato in {run_dir.name}: {e}")

    # prom_pre.json / prom_post.json
    for pname in ('prom_pre', 'prom_post'):
        pp = run_dir / f'{pname}.json'
        if pp.is_file():
            try:
                data = json.loads(pp.read_text(encoding='utf-8'))
                setattr(run, pname, data if isinstance(data, list) else [data])
            except json.JSONDecodeError:
                pass

    # Provenienza (harness schema-2): configurazione effettiva di DMOS e
    # stato del testbed prima/dopo il run. Assenti nei run legacy.
    for fname, attr in (('dmos_config.json', 'dmos_config'),
                        ('env_state_pre.json', 'env_state_pre'),
                        ('env_state_post.json', 'env_state_post')):
        fp = run_dir / fname
        if fp.is_file():
            try:
                setattr(run, attr, json.loads(fp.read_text(encoding='utf-8')))
            except json.JSONDecodeError as e:
                print(f"  [warn] {fname} malformato in {run_dir.name}: {e}")

    # Timeline scheduler+cluster. Formato attuale: dmos_live.jsonl prodotto da
    # collect_dmos_live.py. Fallback al vecchio timeline.csv (run pre-2026).
    if load_timeline:
        jp = run_dir / 'dmos_live.jsonl'
        cp = run_dir / 'timeline.csv'
        if jp.is_file():
            run.timeline_df = _load_dmos_live_jsonl(jp)
        elif cp.is_file():
            run.timeline_df = _load_timeline_csv(cp)

    return run


def _load_timeline_csv(path: Path) -> Optional[pd.DataFrame]:
    """Loader retro-compatibile per i vecchi timeline.csv (Prometheus recorder)."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        t0 = df['ts'].min()
        df['elapsed_s'] = df['ts'] - t0
        for col in ('replicas', 'cpu_rate', 'hubble_reqs_rate', 'memory_bytes'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['phase'] = df['elapsed_s'].apply(assign_phase)
        return df
    except Exception as e:
        print(f"  [warn] timeline.csv error in {path.parent.name}: {e}")
        return None


def compute_cluster_phase_stats(run_dir: Path,
                                slo_ms: float = 2000.0,
                                force_rescan: bool = False) -> Dict:
    """
    Per ogni (cluster, phase) calcola: n, p50, p95, p99, n_violations,
    pct_violations (>slo_ms). Scansiona `k6_events.json` una volta, salva
    cache JSON in `<run_dir>/cluster_phase_stats.json`.

    Usato dalle figure supplementari (Fig 5, 7, 8) per inquadrare il segnale
    per-cluster × per-phase invece che aggregato globale.

    Returns: {
        'slo_ms': float,
        'cells': {
            '<cluster>|<phase>': {
                'cluster': str, 'phase': str, 'n': int,
                'p50': float, 'p95': float, 'p99': float,
                'n_violations': int, 'pct_violations': float
            }, ...
        }
    }
    """
    cache_path = run_dir / 'cluster_phase_stats.json'
    events_path = run_dir / 'k6_events.json'

    # Cache hit: stessa soglia SLO e mtime ok
    if not force_rescan and cache_path.is_file() and events_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            if (abs(cached.get('slo_ms', 0) - slo_ms) < 0.5 and
                    cache_path.stat().st_mtime >= events_path.stat().st_mtime):
                return cached
        except Exception:
            pass

    if not events_path.is_file():
        return {'slo_ms': slo_ms, 'cells': {}}

    needle = b'"metric":"http_req_duration"'
    accum: Dict[Tuple[str, str], List[float]] = {}
    n_viol: Dict[Tuple[str, str], int] = {}

    print(f"  [cphase] parsing {events_path.parent.name}/k6_events.json ...",
          flush=True)

    with events_path.open('rb') as f:
        for line in f:
            if needle not in line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get('type') != 'Point':
                continue
            data = ev.get('data', {})
            try:
                val = float(data.get('value', 0.0))
            except (TypeError, ValueError):
                continue
            tags = data.get('tags', {})
            cluster = tags.get('cluster', 'unknown')
            phase = tags.get('phase', 'unknown')
            key = (cluster, phase)
            accum.setdefault(key, []).append(val)
            if val > slo_ms:
                n_viol[key] = n_viol.get(key, 0) + 1

    cells: Dict[str, dict] = {}
    for (cluster, phase), vals in accum.items():
        arr = np.asarray(vals, dtype=float)
        if len(arr) == 0:
            continue
        nv = n_viol.get((cluster, phase), 0)
        cells[f"{cluster}|{phase}"] = {
            'cluster':        cluster,
            'phase':          phase,
            'n':              int(len(arr)),
            'p50':            float(np.percentile(arr, 50)),
            'p95':            float(np.percentile(arr, 95)),
            'p99':            float(np.percentile(arr, 99)),
            'n_violations':   int(nv),
            'pct_violations': float(100.0 * nv / len(arr)),
        }

    result = {'slo_ms': slo_ms, 'cells': cells}
    try:
        cache_path.write_text(json.dumps(result, indent=2))
    except Exception as e:
        print(f"  [cphase] warning: cache save failed: {e}")
    return result


def load_per_service_per_cluster(path: Path) -> Optional[pd.DataFrame]:
    """
    Carica `dmos_live.jsonl` mantenendo il dettaglio per-servizio: ogni riga
    e' un (snapshot, service, cluster) con `ready_pods`. Usato dalla heatmap
    di Fig 1 per mostrare DOVE ogni servizio viene piazzato (asimmetria di
    placement che il sum-over-services nascondeva).

    Colonne: ts, elapsed_s, service, cluster, replicas, phase
    """
    rows: List[dict] = []
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('{'):
                    continue
                try:
                    snap = json.loads(line)
                except json.JSONDecodeError:
                    continue
                elapsed = float(snap.get('elapsed_s', 0.0) or 0.0)
                try:
                    ts = pd.to_datetime(snap.get('timestamp')).timestamp()
                except Exception:
                    ts = elapsed
                services = snap.get('services', {}) or {}
                for svc_name, svc in services.items():
                    clusters = svc.get('clusters') or {}
                    for cl, cd in clusters.items():
                        rp = cd.get('ready_pods')
                        if rp is None:
                            rp = cd.get('current_replicas', 0)
                        rows.append({
                            'ts':         ts,
                            'elapsed_s':  elapsed,
                            'service':    svc_name,
                            'cluster':    cl,
                            'replicas':   int(rp or 0),
                            'phase':      assign_phase(elapsed),
                        })
    except Exception as e:
        print(f"  [warn] dmos_live.jsonl error in {path.parent.name}: {e}")
        return None
    if not rows:
        return None
    return pd.DataFrame(rows)


def _load_dmos_live_jsonl(path: Path) -> Optional[pd.DataFrame]:
    """
    Converte dmos_live.jsonl (un JSON-object per riga, prodotto da
    collect_dmos_live.py) in un DataFrame long-format: una riga per
    (snapshot, cluster), con le colonne attese dai plot del paper:

        ts, elapsed_s, cluster, replicas, cpu_rate, hubble_reqs_rate,
        memory_bytes, phase

    `replicas` = somma dei pod *ready* (campo k8s `ready_pods`) su tutti i
    servizi del cluster. Si usa ready_pods e non il gauge DMOS
    `current_replicas` perche' quest'ultimo resta 0 nello scenario dmos-off
    (DMOS non gira → nessuno popola il gauge), mentre ready_pods e' letto
    direttamente da Kubernetes ed e' valido per tutti gli scenari.
    """
    rows = []
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('{'):
                    continue
                try:
                    snap = json.loads(line)
                except json.JSONDecodeError:
                    continue
                elapsed = float(snap.get('elapsed_s', 0.0) or 0.0)
                try:
                    ts = pd.to_datetime(snap.get('timestamp')).timestamp()
                except Exception:
                    ts = elapsed
                services = snap.get('services', {}) or {}
                clusters_blk = snap.get('clusters', {}) or {}
                names = set(clusters_blk.keys())
                for svc in services.values():
                    names.update((svc.get('clusters') or {}).keys())
                for cl in sorted(names):
                    reps = 0
                    mem_mb = 0.0
                    for svc in services.values():
                        cd = (svc.get('clusters') or {}).get(cl)
                        if not cd:
                            continue
                        rp = cd.get('ready_pods')
                        if rp is None:
                            rp = cd.get('current_replicas', 0)
                        reps += int(rp or 0)
                        mem_mb += float(cd.get('memory_mb') or 0.0)
                    cblk = clusters_blk.get(cl, {}) or {}
                    rows.append({
                        'ts':               ts,
                        'elapsed_s':        elapsed,
                        'cluster':          cl,
                        'replicas':         reps,
                        'cpu_rate':         cblk.get('node_cpu_pct'),
                        'hubble_reqs_rate': cblk.get('hubble_http_rps'),
                        'memory_bytes':     mem_mb * 1e6,
                        'phase':            assign_phase(elapsed),
                    })
    except Exception as e:
        print(f"  [warn] dmos_live.jsonl error in {path.parent.name}: {e}")
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for col in ('replicas', 'cpu_rate', 'hubble_reqs_rate', 'memory_bytes'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ── Stat helpers ────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray,
                 n_iter: int = 1000,
                 ci: float = 0.95,
                 seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap percentile CI per la media di `values`.

    Returns: (mean, ci_low, ci_high)
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float('nan'), float('nan'), float('nan'))
    if len(values) == 1:
        return (float(values[0]), float(values[0]), float(values[0]))

    rng = np.random.default_rng(seed)
    means = np.empty(n_iter, dtype=float)
    n = len(values)
    for i in range(n_iter):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    alpha = 1.0 - ci
    low, high = np.percentile(means, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return (float(values.mean()), float(low), float(high))


def welch_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Welch's t-test (varianze diverse, campioni indipendenti).
    Returns: (t_statistic, p_value)
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy non installato. pip install scipy")
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return (float('nan'), float('nan'))
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return (float(t), float(p))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d (effect size) tra due campioni.

    Convenzione: d > 0 se mean(a) > mean(b).
    Magnitudine: 0.2 small, 0.5 medium, 0.8 large.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float('nan')
    return float((a.mean() - b.mean()) / pooled)


def cohen_d_label(d: float) -> str:
    """Etichetta qualitativa per Cohen's d."""
    if np.isnan(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


# ── Phase / binning helpers ─────────────────────────────────────────────────

def assign_phase(elapsed_s: float) -> str:
    """Restituisce il nome della fase per un timestamp relativo (in secondi)."""
    for t0, t1, name, _ in PHASE_DEFS:
        if t0 <= elapsed_s < t1:
            return name
    return 'cooldown'   # fallback per t >= 900


def bin_timeline(df: pd.DataFrame, bin_sec: int = BIN_SEC_DEFAULT,
                 agg: Dict[str, str] = None) -> pd.DataFrame:
    """
    Bin di una timeline df per cluster.

    Args:
        df: DataFrame con colonne ['elapsed_s', 'cluster', 'replicas', 'cpu_rate', ...]
        bin_sec: ampiezza del bin in secondi
        agg: mapping colonna → funzione di aggregazione (default: mean)

    Returns: DataFrame con colonne ['bin_s', 'cluster', '<metric>', ...]
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if agg is None:
        agg = {col: 'mean' for col in ['replicas', 'cpu_rate', 'hubble_reqs_rate', 'memory_bytes']
               if col in df.columns}
    work = df.copy()
    work['bin_s'] = (work['elapsed_s'] // bin_sec * bin_sec).astype(int)
    grouped = work.groupby(['bin_s', 'cluster']).agg(agg).reset_index()
    grouped['phase'] = grouped['bin_s'].apply(assign_phase)
    return grouped


# ── Misc utilities ──────────────────────────────────────────────────────────

def fmt_ms(x: float, digits: int = 0) -> str:
    """Format ms with appropriate unit suffix."""
    if np.isnan(x):
        return "n/a"
    if x >= 1000:
        return f"{x/1000:.{digits}f}s" if digits else f"{x:.0f}ms"
    return f"{x:.{digits}f}ms" if digits else f"{x:.0f}ms"


def fmt_pct(x: float, digits: int = 2) -> str:
    """Format probability/rate as percentage."""
    if np.isnan(x):
        return "n/a"
    return f"{x*100:.{digits}f}%"


def fmt_ci(mean: float, low: float, high: float, unit: str = '') -> str:
    """Format mean ± CI95 as 'mean [low, high]'."""
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.1f}{unit} [{low:.1f}, {high:.1f}]"


def shade_phases(ax, x_unit: str = 'min', y_frac_label: float = 0.97):
    """
    Aggiunge bande colorate per ogni fase + etichetta in alto.
    `x_unit` puo' essere 'min' (default) o 's'.
    """
    div = 60.0 if x_unit == 'min' else 1.0
    for (t0, t1, name, color) in PHASE_DEFS:
        ax.axvspan(t0 / div, t1 / div, alpha=0.18, color=color, zorder=0)
        ax.text((t0 + t1) / 2 / div, y_frac_label, name,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=8, alpha=0.6, fontstyle='italic')


def save_figure(fig, out_dir: Path, basename: str,
                formats: Tuple[str, ...] = ('pdf', 'png')) -> List[Path]:
    """Salva la figure in tutti i formati richiesti. Ritorna i path scritti."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        p = out_dir / f"{basename}.{fmt}"
        fig.savefig(p, dpi=TARGET_DPI, format=fmt)
        written.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return written


def print_group_summary(groups: Dict[str, ScenarioGroup]):
    """Stampa una tabella riassuntiva del groups dict."""
    print(f"\n{'─' * 70}")
    print(f"  Scenarios discovered: {len(groups)}")
    print(f"{'─' * 70}")
    for scenario, group in groups.items():
        label = SCENARIO_LABELS.get(scenario, scenario)
        print(f"  {scenario:15s} ({label:30s}): {group.n_repetitions} run(s)")
        for r in group.runs:
            tline_n = 0 if r.timeline_df is None else len(r.timeline_df)
            print(f"    rep #{r.repetition}: {r.run_dir.name}  (timeline={tline_n} rows)")
    print(f"{'─' * 70}\n")
