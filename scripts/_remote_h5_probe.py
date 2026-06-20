from __future__ import annotations
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"; PORT = 12264
USER = "wuxu"; PASSWORD = "Wuxu@96885"
REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"

SCRIPT = """
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
import h5py
for p in ('/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR_fixed.h5ad',
          '/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad'):
    print('---', p)
    try:
        with h5py.File(p, 'r') as f:
            print('keys:', list(f.keys()))
            if 'obs' in f:
                obs = f['obs']
                print('obs keys:', list(obs.keys())[:30])
            if 'X' in f:
                print('X type:', type(f['X']).__name__)
            if 'layers' in f:
                print('layers:', list(f['layers'].keys()))
    except Exception as e:
        print('ERR:', type(e).__name__, str(e)[:200])
"""


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    cmd = f"HDF5_USE_FILE_LOCKING=FALSE {REMOTE_PY} - <<'PY'\n{SCRIPT}\nPY\n"
    _, o, e = c.exec_command(cmd, timeout=180)
    print(o.read().decode(errors="replace"))
    err = e.read().decode(errors="replace").rstrip()
    if err:
        print("[stderr]", err)
    c.close()


if __name__ == "__main__":
    main()
