from __future__ import annotations
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"; PORT = 12264
USER = "wuxu"; PASSWORD = "Wuxu@96885"
REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"

SCRIPT = r"""
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
import h5py, collections

P = '/home/wuxu/jupyter/MJ/newh5ad/acc_new.h5ad'
TARGETS = [
    'celltype', 'celltype_new', 'chondrocyte_type', 'chondrocyte-type',
    'chongdrocyte_subtype', 'chongdrocyte_subtype_cn',
    'current_harmony_leiden_r0_6_annot', 'current_harmony_leiden_r0_6_annot_cn',
    'current_harmony_leiden_r0_7_annot', 'current_harmony_leiden_r0_7_annot_cn',
    'cartilage_type', 'cartilage_cgrm_label', 'general_celltype', 'rare_terminal_program',
]
with h5py.File(P, 'r') as f:
    obs = f['obs']
    for k in TARGETS:
        if k not in obs:
            print(f'== {k}: MISSING')
            continue
        node = obs[k]
        if hasattr(node, 'keys') and 'categories' in node:
            cats = node['categories'][()]
            codes = node['codes'][()]
            cats = [c.decode() if isinstance(c, bytes) else c for c in cats]
            cnt = collections.Counter(codes.tolist())
            print(f'== {k}: {len(cats)} cats')
            rows = sorted([(cnt.get(i, 0), i, n) for i, n in enumerate(cats)], reverse=True)
            for c, i, n in rows[:25]:
                print(f'   {c:>8d}  {i:>3d}  {n}')
        else:
            print(f'== {k}: not categorical')
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
    if err:
        print('[stderr]', err)
    c.close()

if __name__ == '__main__':
    main()
