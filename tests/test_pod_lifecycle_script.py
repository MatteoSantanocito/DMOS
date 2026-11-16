"""
experiments/apply_pod_lifecycle.sh: preStop/grace on the entry service and on
every service DMOS co-scales (they are removed by its scale-downs).
"""
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "experiments" / "apply_pod_lifecycle.sh"
BASH = shutil.which("bash")


def script_services():
    text = SCRIPT.read_text(encoding="utf-8")
    m = re.search(r"^SERVICES=\(([^)]*)\)", text, re.M | re.S)
    assert m, "SERVICES=( ... ) not found"
    return m.group(1).split()


def test_every_scaled_service_gets_the_lifecycle():
    app = yaml.safe_load((REPO / "config" / "services.yaml").read_text(encoding="utf-8"))["application"]
    needed = {app["entry_service"], *app["co_scaled_services"]}
    assert needed <= set(script_services()), f"missing: {sorted(needed - set(script_services()))}"


FAKE_KUBECTL = r"""#!/usr/bin/env bash
args="$*"
args="${args//$'\n'/ }"
echo "$args" >> "$KUBECTL_LOG"
if [[ "$args" == *"containers[*].name"* ]]; then
  [[ "$args" == *" adservice "* && -n "${NO_SERVER_FOR_AD:-}" ]] && { echo "main"; exit 0; }
  echo "server"; exit 0
fi
exit 0
"""


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_patches_all_services_on_all_clusters_and_skips_without_server(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(FAKE_KUBECTL, encoding="utf-8")
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IEXEC)
    kc_dir = tmp_path / "kc"
    kc_dir.mkdir()
    for n in ("01", "02", "03", "04", "05", "06"):
        (kc_dir / f"kubeconfig-ms-cluster-{n}").write_text("x", encoding="utf-8")
    log = tmp_path / "kubectl.log"
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
               DMOS_KUBECONFIG_DIR=str(kc_dir), KUBECTL_LOG=str(log), NO_SERVER_FOR_AD="1")
    env.pop("DMOS_LIFECYCLE_SERVICES", None)
    out = subprocess.run([BASH, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
    calls = log.read_text(encoding="utf-8").splitlines()
    patched = [c for c in calls if " patch deployment " in c]
    services = script_services()
    assert len(patched) == 6 * (len(services) - 1)            # adservice skipped everywhere
    assert not any(" adservice " in c for c in patched)
    assert all('"name": "server"' in c and '"seconds": 15' in c
               and '"terminationGracePeriodSeconds": 30' in c for c in patched)
    assert "[skip] adservice" in out.stdout
    rollouts = [c for c in calls if " rollout status deploy/" in c]
    assert len(rollouts) == 6 * len(services)
    assert out.returncode != 0                                 # a skip counts as a failure

    log.write_text("", encoding="utf-8")
    env.pop("NO_SERVER_FOR_AD")
    out = subprocess.run([BASH, str(SCRIPT), "--no-wait"], env=env, capture_output=True,
                         text=True, timeout=60)
    calls = log.read_text(encoding="utf-8").splitlines()
    assert sum(" patch deployment " in c for c in calls) == 6 * len(services)
    assert not any("rollout status" in c for c in calls)
    assert out.returncode == 0, out.stdout + out.stderr
