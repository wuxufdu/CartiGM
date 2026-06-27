"""Push _remote_homeostatic_panel_remote.py to the remote host, run it, and
pull the resulting JSON back to F:/cartifm/outputs/EBR_p4_remote/calibration/.
"""
from __future__ import annotations

from pathlib import Path

import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"
PORT = 12264
USER = "wuxu"
PASSWORD = "Wuxu@96885"

REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"
REMOTE_SCRIPT_PATH = "/tmp/_homeostatic_panel_remote.py"
REMOTE_OUT = "/tmp/homeostatic_panel.json"
REMOTE_LOG = "/tmp/homeostatic_panel.log"
REMOTE_H5 = "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad"

LOCAL_SCRIPT = Path("F:/cartifm/CartiGM/scripts/_remote_homeostatic_panel_remote.py")
LOCAL_OUT_DIR = Path("F:/cartifm/outputs/EBR_p4_remote/calibration")
LOCAL_OUT = LOCAL_OUT_DIR / "homeostatic_panel.json"
LOCAL_LOG = LOCAL_OUT_DIR / "homeostatic_panel.log"


def main() -> None:
    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("connecting", HOST)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)

    sftp = c.open_sftp()
    print("upload", LOCAL_SCRIPT, "->", REMOTE_SCRIPT_PATH)
    sftp.put(str(LOCAL_SCRIPT), REMOTE_SCRIPT_PATH)
    sftp.close()

    cmd = (
        f"EBR_H5={REMOTE_H5} PANEL_OUT={REMOTE_OUT} "
        f"{REMOTE_PY} {REMOTE_SCRIPT_PATH} > {REMOTE_LOG} 2>&1; "
        f"echo DONE_RC=$?"
    )
    print("$", cmd)
    _, o, e = c.exec_command(cmd, timeout=1800)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    print(out)
    if err.strip():
        print("[stderr]", err)

    sftp = c.open_sftp()
    print("download", REMOTE_LOG, "->", LOCAL_LOG)
    sftp.get(REMOTE_LOG, str(LOCAL_LOG))
    print("download", REMOTE_OUT, "->", LOCAL_OUT)
    sftp.get(REMOTE_OUT, str(LOCAL_OUT))
    sftp.close()
    c.close()
    print("ok ->", LOCAL_OUT)


if __name__ == "__main__":
    main()
