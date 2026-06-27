"""Run cartigsfm scgpt-pretrain on the remote server (5070 GPU)."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"; PORT = 12264
USER = "wuxu"; PASSWORD = "Wuxu@96885"

REMOTE_BASE = "/home/wuxu/cartigsfm_remote"
REMOTE_TAR = "/tmp/cartigsfm_pkg.tar.gz"
REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"
REMOTE_OUTDIR = "/home/wuxu/cartigsfm_remote/scgpt_pretrain"
REMOTE_LOG = "/tmp/scgpt_pretrain.log"
ACC_H5 = "/home/wuxu/jupyter/MJ/newh5ad/acc_new.h5ad"

LOCAL_ROOT = Path("F:/cartifm/CartiGM")
LOCAL_OUT = Path("F:/cartifm/outputs/scgpt_pretrain_remote")

# 80k cells, 2k HVG, 2000 steps, batch 256, d_model=256, 6 layers, lr=8e-4
# Using user-provided defaults from cli.py (cmd_scgpt_pretrain).
N_CELLS = "80000"
N_HVG = "2000"
N_STEPS = "2000"
BATCH_SIZE = "256"
D_MODEL = "256"
N_LAYERS = "6"
N_HEADS = "8"
LR = "8e-4"


def _build_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in (LOCAL_ROOT / "cartigsfm").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                rel = path.relative_to(LOCAL_ROOT)
                tar.add(str(path), arcname=str(rel).replace("\\", "/"))
        for extra in ("pyproject.toml", "README.md"):
            p = LOCAL_ROOT / extra
            if p.is_file():
                tar.add(str(p), arcname=extra)
    buf.seek(0)
    return buf.read()


def main() -> None:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    print("Connecting", HOST)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)

    print("Building tarball + uploading")
    tar_bytes = _build_tarball()
    sftp = c.open_sftp()
    with sftp.open(REMOTE_TAR, "wb") as f:
        f.write(tar_bytes)
    sftp.close()
    _, o, _ = c.exec_command(
        f"mkdir -p {REMOTE_BASE} {REMOTE_OUTDIR} && "
        f"tar xzf {REMOTE_TAR} -C {REMOTE_BASE}", timeout=120)
    print(o.read().decode())

    # Probe torch + GPU
    _, o, e = c.exec_command(
        f"{REMOTE_PY} -c \"import torch;print('torch',torch.__version__);"
        f"print('cuda',torch.cuda.is_available());"
        f"print('device',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"",
        timeout=60)
    print(o.read().decode(), e.read().decode())

    cmd = (
        f"cd {REMOTE_BASE} && "
        f"PYTHONPATH={REMOTE_BASE} HDF5_USE_FILE_LOCKING=FALSE "
        f"CUDA_VISIBLE_DEVICES=0 "
        f"{REMOTE_PY} -m cartigsfm.cli scgpt-pretrain "
        f"--h5ad {ACC_H5} --outdir {REMOTE_OUTDIR} "
        f"--n-cells {N_CELLS} --n-hvg {N_HVG} --n-steps {N_STEPS} "
        f"--batch-size {BATCH_SIZE} --d-model {D_MODEL} "
        f"--n-layers {N_LAYERS} --n-heads {N_HEADS} --lr {LR} "
        f"> {REMOTE_LOG} 2>&1; echo DONE_RC=$?"
    )
    print("$", cmd)
    _, o, e = c.exec_command(cmd, timeout=21600)
    print(o.read().decode(errors="replace"))
    err = e.read().decode(errors="replace").rstrip()
    if err:
        print("[stderr]", err)

    print("\n=== tail log ===")
    _, o, _ = c.exec_command(f"tail -80 {REMOTE_LOG}", timeout=30)
    print(o.read().decode(errors="replace"))
    _, o, _ = c.exec_command(f"ls -lh {REMOTE_OUTDIR}/", timeout=30)
    print(o.read().decode(errors="replace"))

    sftp = c.open_sftp()
    for name in ("scgpt_small.pt", "config.json", "pretrain_summary.json",
                 "gene_order.tsv", "train_loss.tsv"):
        try:
            sftp.get(f"{REMOTE_OUTDIR}/{name}", str(LOCAL_OUT / name))
            print("pulled", name)
        except Exception as exc:
            print("skip", name, exc)
    try:
        sftp.get(REMOTE_LOG, str(LOCAL_OUT / "scgpt_pretrain.log"))
    except Exception as exc:
        print("skip log", exc)
    sftp.close()
    c.close()
    print("ok ->", LOCAL_OUT)


if __name__ == "__main__":
    main()
