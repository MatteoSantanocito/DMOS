"""
scripts/setup-netem-and-ping.sh: the k6 client is exempt from the netem delay
(users close to the ingress they enter), unless DMOS_NETEM_CLIENT_LOCAL=0.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "setup-netem-and-ping.sh"
BASH = shutil.which("bash")


def _function(name):
    text = SCRIPT.read_text(encoding="utf-8")
    m = re.search(rf"^{name}\(\) {{\n.*?^}}\n", text, re.S | re.M)
    assert m, f"{name} not found in {SCRIPT.name}"
    return m.group(0)


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("env,expected", [
    ({}, "192.168.0.140"),                                          # .k6.remote of test-env.json
    ({"DMOS_K6_REMOTE": "me@10.0.0.5"}, "10.0.0.5"),
    ({"DMOS_K6_REMOTE": "me@10.0.0.5:22"}, "10.0.0.5"),
    ({"DMOS_NETEM_CLIENT_IPS": "1.2.3.4 5.6.7.8"}, "1.2.3.4 5.6.7.8"),
    ({"DMOS_NETEM_CLIENT_LOCAL": "0"}, ""),                         # old geography
    ({"DMOS_K6_REMOTE": "local"}, ""),                              # k6 on the laptop: IP unknown
])
def test_netem_client_ips(tmp_path, env, expected):
    if not env and shutil.which("jq") is None:
        pytest.skip("jq not available (reads .k6.remote of test-env.json)")
    env_json = tmp_path / "test-env.json"
    env_json.write_text(json.dumps({"k6": {"remote": "ubuntu@192.168.0.140"}}), encoding="utf-8")
    script = f"set -euo pipefail\n{_function('netem_client_ips')}\nnetem_client_ips || true\n"
    run_env = {k: v for k, v in os.environ.items()
               if not k.startswith(("DMOS_K6_REMOTE", "DMOS_NETEM_"))}
    run_env.update(env, ENV_JSON=str(env_json))
    out = subprocess.run([BASH, "-c", script], env=run_env, capture_output=True, text=True,
                         timeout=20)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == expected


def test_apply_netem_exempts_the_clients():
    body = _function("apply_netem_cluster")
    assert 'clients="$(netem_client_ips' in body
    assert 'own="$(echo $ips $infra $clients)"' in body
