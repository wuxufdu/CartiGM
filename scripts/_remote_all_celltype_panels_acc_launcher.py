"""Push acc DE script + run on remote, pull JSON."""
from __future__ import annotations

from pathlib import Path
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"
PORT = 12264
USER = "wuxu"
PASSWORD = "Wuxu@96885"

REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"
REMOTE_SCRIPT = "/tmp/_all_celltype_panels_acc_remote.py"
REMOTE_OUT = "/tmp/all_celltype_panels_acc.json"
REMOTE_LOG = "/tmp/all_celltype_panels_acc.log"
REMOTE_H5 = "/home/wuxu/jupyter/MJ/newh5ad/acc_new.h5ad"
LABEL_COL = "celltype_new"
MAX_PER_GROUP = "5000"

LOCAL_SCRIPT = Path("F:/cartifm/CartiGM/scripts/_remote_all_celltype_panels_acc_remote.py")
LOCAL_OUT_DIR = Path("F:/cartifm/outputs/EBR_p4_remote/calibration")
LOCAL_OUT = LOCAL_OUT_DIR / "all_celltype_panels_acc.json"
LOCAL_LOG = LOCAL_OUT_DIR / "all_celltype_panels_acc.log"


def main() -> None:
    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    sftp = c.open_sftp()
    sftp.put(str(LOCAL_SCRIPT), REMOTE_SCRIPT)
    sftp.close()
    cmd = (
        f"HDF5_USE_FILE_LOCKING=FALSE "
        f"ACC_H5={REMOTE_H5} "
        f"LABEL_COL={LABEL_COL} "
        f"MAX_PER_GROUP={MAX_PER_GROUP} "
        f"PANEL_OUT={REMOTE_OUT} "
        f"{REMOTE_PY} {REMOTE_SCRIPT} > {REMOTE_LOG} 2>&1; "
        f"echo DONE_RC=$?"
    )
    print("$", cmd)
    _, o, e = c.exec_command(cmd, timeout=3600)
    print(o.read().decode(errors="replace"))
    err = e.read().decode(errors="replace")
    if err.strip():
        print("[stderr]", err)
    sftp = c.open_sftp()
    try:
        sftp.get(REMOTE_LOG, str(LOCAL_LOG))
    except Exception as exc:
        print("log pull err", exc)
    try:
        sftp.get(REMOTE_OUT, str(LOCAL_OUT))
    except Exception as exc:
        print("json pull err", exc)
    sftp.close()
    c.close()
    print("ok ->", LOCAL_OUT)


if __name__ == "__main__":
    main()
