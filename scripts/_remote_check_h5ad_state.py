from __future__ import annotations
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"; PORT = 12264
USER = "wuxu"; PASSWORD = "Wuxu@96885"


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    for cmd in (
        "date",
        "ls -lh /home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad /home/wuxu/jupyter/MJ/newh5ad/EBL/EBR_fixed.h5ad",
        "lsof /home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad /home/wuxu/jupyter/MJ/newh5ad/EBL/EBR_fixed.h5ad 2>&1",
        "lsof -p $(pgrep -f 'jupyter-notebook' | head -1) 2>/dev/null | grep -E 'EBR|h5ad' | head -10",
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
