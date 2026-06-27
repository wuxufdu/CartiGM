"""Push the all-celltype DE script to the remote host, run it, pull the JSON."""
from __future__ import annotations

from pathlib import Path
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"
PORT = 12264
USER = "wuxu"
PASSWORD = "Wuxu@96885"

REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"
REMOTE_SCRIPT = "/tmp/_all_celltype_panels_remote.py"
REMOTE_OUT = "/tmp/all_celltype_panels.json"
REMOTE_LOG = "/tmp/all_celltype_panels.log"
REMOTE_H5 = "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR_fixed.h5ad"

LOCAL_SCRIPT = Path("F:/cartifm/CartiGM/scripts/_remote_all_celltype_panels_remote.py")
LOCAL_OUT_DIR = Path("F:/cartifm/outputs/EBR_p4_remote/calibration")
LOCAL_OUT = LOCAL_OUT_DIR / "all_celltype_panels.json"
LOCAL_LOG = LOCAL_OUT_DIR / "all_celltype_panels.log"


def main() -> None:
    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    sftp = c.open_sftp()
    sftp.put(str(LOCAL_SCRIPT), REMOTE_SCRIPT)
    sftp.close()
    cmd = (f"HDF5_USE_FILE_LOCKING=FALSE "
           f"EBR_H5={REMOTE_H5} PANEL_OUT={REMOTE_OUT} "
           f"{REMOTE_PY} {REMOTE_SCRIPT} > {REMOTE_LOG} 2>&1; "
           f"echo DONE_RC=$?")
    print("$", cmd)
    _, o, e = c.exec_command(cmd, timeout=1800)
    print(o.read().decode(errors="replace"))
    err = e.read().decode(errors="replace")
    if err.strip():
        print("[stderr]", err)
    sftp = c.open_sftp()
    sftp.get(REMOTE_LOG, str(LOCAL_LOG))
    sftp.get(REMOTE_OUT, str(LOCAL_OUT))
    sftp.close()
    c.close()
    print("ok ->", LOCAL_OUT)


if __name__ == "__main__":
    main()
