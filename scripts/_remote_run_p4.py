"""Push the cartigsfm package + dictionary to the remote host and run
`cartigsfm p4-project` against the user-annotated EBR.h5ad. The full pipeline
stays remote; only the resulting TSVs and the markdown report are pulled back
to ``F:/cartifm/outputs/EBR_p4_remote``.
"""
from __future__ import annotations

import io
import os
import sys
import tarfile
import time
from pathlib import Path

import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"
PORT = 12264
USER = "wuxu"
PASSWORD = "Wuxu@96885"

REMOTE_BASE = "/home/wuxu/cartigsfm_remote"
REMOTE_TAR = "/tmp/cartigsfm_pkg.tar.gz"
REMOTE_OUT = f"{REMOTE_BASE}/p4_outdir"
REMOTE_LOG = f"{REMOTE_BASE}/p4_run.log"
REMOTE_H5 = "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad"
REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"

LOCAL_ROOT = Path("F:/cartifm/CartiGM")
LOCAL_OUT = Path("F:/cartifm/outputs/EBR_p4_remote")


def _build_tarball() -> bytes:
    """Pack the cartigsfm package + dictionary + RAG into a tar.gz (in-memory)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # The cartigsfm package itself
        for path in (LOCAL_ROOT / "cartigsfm").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                rel = path.relative_to(LOCAL_ROOT)
                tar.add(str(path), arcname=str(rel).replace("\\", "/"))
        # pyproject so we can pip install -e
        for extra in ("pyproject.toml", "README.md"):
            p = LOCAL_ROOT / extra
            if p.is_file():
                tar.add(str(p), arcname=extra)
    buf.seek(0)
    return buf.read()


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 600, label: str | None = None) -> str:
    if label:
        print(f"$ # {label}")
    print(f"$ {cmd}")
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors="replace").rstrip()
    err = e.read().decode(errors="replace").rstrip()
    if out:
        print(out)
    if err:
        print("[stderr]", err)
    return out


def main():
    print("Connecting", HOST)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)

    print("Building tarball")
    tar_bytes = _build_tarball()
    print(f"  tarball size: {len(tar_bytes)/1024:.1f} KB")

    sftp = c.open_sftp()
    with sftp.open(REMOTE_TAR, "wb") as f:
        f.write(tar_bytes)
    sftp.close()

    run(c, f"mkdir -p {REMOTE_BASE} {REMOTE_OUT}", label="prepare dirs")
    run(c, f"tar xzf {REMOTE_TAR} -C {REMOTE_BASE}", label="extract")
    run(c, f"ls {REMOTE_BASE}", label="layout check")

    # Make sure CLI deps exist in the chosen env. squidp311 has anndata 0.12.10
    # but we need the cartigsfm package + its lightweight deps.
    run(c, f"{REMOTE_PY} -c 'import scipy, numpy, pandas, anndata; "
           "print(scipy.__version__, numpy.__version__, pandas.__version__, anndata.__version__)'",
        label="env probe")

    # Use PYTHONPATH so we don't pollute the env with `pip install -e`.
    env = f"PYTHONPATH={REMOTE_BASE}"
    cli = f"{env} {REMOTE_PY} -m cartigsfm.cli"
    run(c, f"{cli} --help 2>&1 | head -20", label="cli help")

    anti_lambda = float(__import__('os').environ.get("P4_ANTI_LAMBDA", "0.5"))
    cmd = (
        f"{cli} p4-project "
        f"--h5ad {REMOTE_H5} "
        f"--outdir {REMOTE_OUT} "
        f"--sample-col batch "
        f"--tissue-col batch "
        f"--cluster-col leiden_res1 "
        f"--celltype-col celltype "
        f"--celltype-regex chondro "
        f"--layer log1p_norm "
        f"--streaming --chunk-size 4000 "
        f"--gene-col gene "
        f"--anti-lambda {anti_lambda}"
    )
    full = f"({cmd}) > {REMOTE_LOG} 2>&1; echo DONE_RC=$?"
    print(f"Launching P4 (this can take several minutes for 32k cells × 29k genes)")
    rc_line = run(c, full, timeout=7200, label="run p4-project")
    print(rc_line)

    print("Tailing log")
    run(c, f"tail -120 {REMOTE_LOG}", label="log tail")

    print("Listing outputs")
    run(c, f"ls -lh {REMOTE_OUT}/tsv/ {REMOTE_OUT}/docs/", label="outputs")

    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    sftp = c.open_sftp()
    for sub in ("tsv", "docs"):
        try:
            files = sftp.listdir(f"{REMOTE_OUT}/{sub}")
        except IOError:
            continue
        (LOCAL_OUT / sub).mkdir(parents=True, exist_ok=True)
        for name in files:
            remote = f"{REMOTE_OUT}/{sub}/{name}"
            local = LOCAL_OUT / sub / name
            print(f"  pulling {remote} -> {local}")
            sftp.get(remote, str(local))
    # also pull the run log
    sftp.get(REMOTE_LOG, str(LOCAL_OUT / "p4_run.log"))
    sftp.close()
    c.close()

    print("Done. Local outputs:", LOCAL_OUT)


if __name__ == "__main__":
    main()
