"""Find recent h5ad files in /home/wuxu and EBL directory."""
from __future__ import annotations
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"
PORT = 12264
USER = "wuxu"
PASSWORD = "Wuxu@96885"


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    for cmd in (
        "ls -lh /home/wuxu/jupyter/MJ/newh5ad/EBL/",
        "ls -lh /home/wuxu/jupyter/MJ/newh5ad/",
        "find /home/wuxu -maxdepth 6 -name 'EBR*.h5ad' -printf '%T@ %s %p\\n' 2>/dev/null | sort -nr | head -20",
        "find /home/wuxu -maxdepth 6 -name '*.h5ad' -mtime -7 -printf '%T@ %s %p\\n' 2>/dev/null | sort -nr | head -30",
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
