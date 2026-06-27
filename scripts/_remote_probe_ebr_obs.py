from __future__ import annotations
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"; PORT = 12264
USER = "wuxu"; PASSWORD = "Wuxu@96885"
REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"

SCRIPT = r"""
import os, collections
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
import h5py
P = '/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad'
with h5py.File(P, 'r') as f:
    print('keys:', list(f.keys()))
    if 'layers' in f:
        print('layers:', list(f['layers'].keys()))
    obs = f['obs']
    cols = list(obs.keys())
    print('obs cols (', len(cols), '):', cols)
    if 'X' in f:
        x = f['X']
        if hasattr(x, 'attrs'):
            print('X enc:', dict(x.attrs))
    for k in cols:
        node = obs[k]
        if hasattr(node, 'keys') and 'categories' in node:
            cats = node['categories'][()]
            cnt = collections.Counter(node['codes'][()].tolist())
            cats = [c.decode() if isinstance(c, bytes) else c for c in cats]
            top = sorted([(cnt.get(i,0), n) for i, n in enumerate(cats)], reverse=True)[:8]
            print(f'  cat {k}: {len(cats)} values, top {top}')
"""

def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    cmd = f"HDF5_USE_FILE_LOCKING=FALSE {REMOTE_PY} - <<'PY'\n{SCRIPT}\nPY\n"
    _, o, e = c.exec_command(cmd, timeout=180)
    print(o.read().decode(errors='replace'))
    err = e.read().decode(errors='replace').rstrip()
    if err: print('[stderr]', err)
    c.close()

if __name__ == '__main__':
    main()
