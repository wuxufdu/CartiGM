"""Diagnose: who is holding EBR.h5ad on the remote host?"""
from __future__ import annotations
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"
PORT = 12264
USER = "wuxu"
PASSWORD = "Wuxu@96885"
REMOTE_H5 = "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad"


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    for cmd in (
        f"ls -lh {REMOTE_H5}",
        f"stat {REMOTE_H5}",
        f"lsof {REMOTE_H5} 2>&1 | head -40",
        "ps -eo pid,etime,cmd | grep -E 'python|jupyter|h5' | grep -v grep | head -40",
    ):
        print("$", cmd)
        _, o, e = c.exec_command(cmd, timeout=120)
        print(o.read().decode(errors="replace"))
        err = e.read().decode(errors="replace").rstrip()
        if err:
            print("[stderr]", err)
    c.close()


if __name__ == "__main__":
    main()
