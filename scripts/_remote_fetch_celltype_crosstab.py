"""Run a tiny remote query to grab the celltype x batch x leiden crosstab
as a small TSV. Avoids any heavy h5ad transfer.
"""
from __future__ import annotations

import paramiko
from pathlib import Path

HOST = "sw2-dynamic.xiyoucloud.pro"
PORT = 12264
USER = "wuxu"
PASSWORD = "Wuxu@96885"

REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"
REMOTE_H5 = "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad"
REMOTE_OUT = "/tmp/ebr_celltype_crosstab.tsv"

LOCAL_OUT = Path("F:/cartifm/outputs/EBR_p4_remote/tsv/p4_celltype_crosstab.tsv")

SCRIPT = f"""
import anndata as ad, pandas as pd
a = ad.read_h5ad('{REMOTE_H5}', backed='r')
obs = a.obs[['batch','celltype','leiden_res1']].astype(str)
ct = obs.groupby(['batch','leiden_res1','celltype']).size().rename('n').reset_index()
ct.to_csv('{REMOTE_OUT}', sep='\\t', index=False)
print('rows:', len(ct))
"""


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    cmd = f"{REMOTE_PY} - <<'PY'\n{SCRIPT}\nPY\n"
    _, o, e = c.exec_command(cmd, timeout=300)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    print(out)
    if err.strip():
        print("[stderr]", err)
    sftp = c.open_sftp()
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(REMOTE_OUT, str(LOCAL_OUT))
    sftp.close()
    c.close()
    print("saved:", LOCAL_OUT)


if __name__ == "__main__":
    main()
