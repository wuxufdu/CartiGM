from __future__ import annotations
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"; PORT = 12264
USER = "wuxu"; PASSWORD = "Wuxu@96885"

def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    cmds = [
        "ls /home/wuxu/miniconda3/envs/",
        "nvidia-smi | head -20",
        "df -h /home /tmp",
        "/home/wuxu/miniconda3/envs/squidp311/bin/python -c 'import torch;print(torch.__version__,torch.version.cuda)' 2>&1 | head -5",
    ]
    for cmd in cmds:
        print('$', cmd)
        _, o, e = c.exec_command(cmd, timeout=30)
        print(o.read().decode())
        err = e.read().decode().rstrip()
        if err: print('[stderr]', err)
        print()
    c.close()

if __name__ == "__main__":
    main()
