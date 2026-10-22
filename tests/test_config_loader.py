"""
Test ConfigLoader — topology-agnostic.

Non assume una topologia specifica (numero/nomi dei cluster): verifica
proprietà strutturali della configurazione caricata, così il test resta
valido quando la fleet cambia (es. 3 cluster legacy -> 6 ms-cluster-0x).

config/local.yaml è gitignored (environment-specific): in un clone fresco
il test fa fallback su config/local.yaml.example, e se manca anche quella
salta con pytest.skip invece di fallire.
"""

import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from src.utils.config_loader import ConfigLoader


def _find_config_dir() -> Path:
    """Config dir: prima relativa alla cwd (comportamento storico del test),
    poi relativa alla root del repo (cwd diverse, es. pytest da altra dir)."""
    cwd_config = Path("config")
    if cwd_config.is_dir():
        return cwd_config
    repo_config = Path(__file__).resolve().parent.parent / "config"
    if repo_config.is_dir():
        return repo_config
    pytest.skip("config/ directory non trovata")


def _loader_from_example(config_dir: Path, tmp_path_factory) -> ConfigLoader:
    """Costruisce un ConfigLoader da config/local.yaml.example.

    Simula il clone fresco: copia i file statici di config/ in una dir
    temporanea e rinomina local.yaml.example -> local.yaml.
    """
    example = config_dir / "local.yaml.example"
    if not example.exists():
        pytest.skip(f"{example} non presente: impossibile fare fallback")
    tmp_config = tmp_path_factory.mktemp("config")
    for f in config_dir.glob("*.yaml"):
        shutil.copy2(f, tmp_config / f.name)
    shutil.copy2(example, tmp_config / "local.yaml")
    return ConfigLoader(config_dir=str(tmp_config))


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    """ConfigLoader sulla config reale; fallback su local.yaml.example."""
    config_dir = _find_config_dir()
    try:
        return ConfigLoader(config_dir=str(config_dir))
    except FileNotFoundError:
        # Clone fresco: local.yaml (gitignored) assente.
        try:
            return _loader_from_example(config_dir, tmp_path_factory)
        except FileNotFoundError as exc:
            pytest.skip(f"config incompleta anche col fallback example: {exc}")


def _assert_cluster_structure(clusters):
    """Verifiche strutturali sui cluster, indipendenti dalla topologia."""
    # DMOS supporta N >= 2 cluster
    assert len(clusters) >= 2, f"attesi >= 2 cluster, trovati {len(clusters)}"

    for name, cluster in clusters.items():
        assert isinstance(name, str) and name
        assert cluster.name == name

        # Campi obbligatori: stringhe non vuote
        assert isinstance(cluster.region, str) and cluster.region
        assert isinstance(cluster.location, str) and cluster.location
        assert isinstance(cluster.ip, str) and cluster.ip
        assert isinstance(cluster.kubeconfig_path, str) and cluster.kubeconfig_path

        # Risorse: tipi e range sensati
        assert isinstance(cluster.resources, dict)
        assert isinstance(cluster.cpu_cores, int) and cluster.cpu_cores >= 1
        assert isinstance(cluster.memory_gb, (int, float)) and cluster.memory_gb > 0

        # Pricing / latenza (derivati dai dict con default nel loader)
        assert isinstance(cluster.cost_per_replica_hour, (int, float))
        assert cluster.cost_per_replica_hour >= 0
        assert isinstance(cluster.baseline_latency_ms, (int, float))
        assert cluster.baseline_latency_ms >= 0

        # Porte valide, se il campo esiste nella versione del loader
        for port_attr in ("prometheus_port", "k8s_api_port"):
            port = getattr(cluster, port_attr, None)
            if port is not None:
                assert isinstance(port, int) and 1 <= port <= 65535, (
                    f"{name}.{port_attr}={port} non è una porta valida"
                )


def test_clusters_structure(config):
    _assert_cluster_structure(config.clusters)
    # get_cluster coerente con il dict
    any_name = next(iter(config.clusters))
    assert config.get_cluster(any_name) is config.clusters[any_name]
    assert config.get_cluster("__nonexistent__") is None


def test_weights_sum_to_one(config):
    assert abs(config.weights.alpha + config.weights.beta +
               config.weights.gamma + config.weights.delta - 1.0) < 1e-6

    # Score weights: lo schema dei campi omega_* varia tra versioni del
    # loader, quindi sommiamo tutti i campi numerici del dataclass.
    score_total = sum(
        v for v in asdict(config.score_weights).values()
        if isinstance(v, (int, float))
    )
    assert abs(score_total - 1.0) < 1e-6


def test_services_structure(config):
    assert len(config.services) > 0

    for name, service in config.services.items():
        assert service.name == name
        assert isinstance(service.namespace, str) and service.namespace
        assert isinstance(service.deployment_name, str) and service.deployment_name
        assert isinstance(service.capacity_req_per_sec, (int, float))
        assert service.capacity_req_per_sec > 0
        assert isinstance(service.min_replicas, int) and service.min_replicas >= 0
        assert isinstance(service.max_replicas, int)
        assert service.max_replicas >= max(service.min_replicas, 1)

        # Capacità per-cluster: definita (>0) per ogni cluster della fleet
        if hasattr(service, "get_capacity_for_cluster"):
            for cluster_name in config.clusters:
                assert service.get_capacity_for_cluster(cluster_name) > 0

    # Il servizio d'ingresso (frontend) deve esistere
    frontend = config.get_service("frontend")
    assert frontend is not None
    assert frontend.namespace


def test_prometheus_config(config):
    assert isinstance(config.prometheus.url, str) and config.prometheus.url
    assert isinstance(config.prometheus.timeout_seconds, int)
    assert config.prometheus.timeout_seconds > 0


def test_example_template_loads(tmp_path_factory):
    """config/local.yaml.example deve produrre una config valida:
    è ciò che vede chi clona il repo senza una local.yaml propria."""
    config_dir = _find_config_dir()
    config = _loader_from_example(config_dir, tmp_path_factory)
    _assert_cluster_structure(config.clusters)
    assert len(config.services) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
