from __future__ import annotations
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"; PORT = 12264
USER = "wuxu"; PASSWORD = "Wuxu@96885"
REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"

SCRIPT = """
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
import h5py
import json

P = '/home/wuxu/jupyter/MJ/newh5ad/acc_new.h5ad'
with h5py.File(P, 'r') as f:
    print('keys:', list(f.keys()))
    obs = f['obs']
    print('obs keys:', list(obs.keys()))
    # try to read 'celltype' as categorical
    if 'celltype' in obs:
        ct = obs['celltype']
        # categorical: codes + categories
        if hasattr(ct, 'keys'):
            print('celltype keys:', list(ct.keys()))
            try:
                cats = ct['categories'][()]
                codes = ct['codes'][()]
                import collections
                bc = [c.decode() if isinstance(c, bytes) else c for c in cats]
                cnt = collections.Counter(codes.tolist())
                print('celltype categories:')
                for i, name in enumerate(bc):
                    print(' ', i, name, cnt.get(i, 0))
            except Exception as e:
                print('cat err', e)
        else:
            print('celltype is dataset shape', ct.shape, 'dtype', ct.dtype)
    if 'batch' in obs:
        bt = obs['batch']
        if hasattr(bt, 'keys'):
            cats = bt['categories'][()]
            print('batch categories:', [c.decode() if isinstance(c, bytes) else c for c in cats])
    if 'layers' in f:
        print('layers:', list(f['layers'].keys()))
    if 'X' in f:
        x = f['X']
        if hasattr(x, 'attrs'):
            print('X encoding:', dict(x.attrs))
        try:
            print('X shape:', x.shape if hasattr(x, 'shape') else 'sparse')
        except: pass
    print('n_obs from _index:', obs['_index'].shape if '_index' in obs else 'NA')
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
