"""Inspect /home/wuxu/jupyter/MJ/newh5ad/EBL/EBR_fixed.h5ad: obs columns +
celltype distribution. We expect the same schema as EBR.h5ad.
"""
from __future__ import annotations
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"
PORT = 12264
USER = "wuxu"
PASSWORD = "Wuxu@96885"
REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"
REMOTE_H5 = "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR_fixed.h5ad"

SCRIPT = f"""
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
import anndata as ad
a = ad.read_h5ad('{REMOTE_H5}', backed='r')
print('shape', a.shape)
print('obs columns:', list(a.obs.columns)[:30])
for col in ('batch','celltype','leiden_res0_5','leiden_res1'):
    if col in a.obs.columns:
        vc = a.obs[col].astype(str).value_counts()
        print(col, ':', dict(vc))
if 'log1p_norm' in a.layers:
    print('layers OK: log1p_norm')
else:
    print('layers:', list(a.layers.keys()))
"""


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    cmd = f"HDF5_USE_FILE_LOCKING=FALSE {REMOTE_PY} - <<'PY'\n{SCRIPT}\nPY\n"
    _, o, e = c.exec_command(cmd, timeout=300)
    print(o.read().decode(errors="replace"))
    err = e.read().decode(errors="replace").rstrip()
    if err:
        print("[stderr]", err)
    c.close()


if __name__ == "__main__":
    main()
