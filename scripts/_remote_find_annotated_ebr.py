from __future__ import annotations
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"; PORT = 12264
USER = "wuxu"; PASSWORD = "Wuxu@96885"
REMOTE_PY = "/home/wuxu/miniconda3/envs/squidp311/bin/python"

# probe a list of candidate h5ad files for celltype column
CANDIDATES = [
    "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad",
    "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR_fixed.h5ad",
    "/home/wuxu/jupyter/MJ/newh5ad/EBL/ear",
    "/home/wuxu/jupyter/MJ/newh5ad/EBL/nose",
    "/home/wuxu/jupyter/MJ/newh5ad/EBL/rib",
]

SCRIPT_FIND = "find /home/wuxu/jupyter/MJ/newh5ad -maxdepth 5 -name '*.h5ad' -printf '%T@ %s %p\\n' 2>/dev/null | sort -nr | head -40"

SCRIPT_PROBE = """
import os, sys
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
import h5py
paths = sys.argv[1:]
for p in paths:
    try:
        with h5py.File(p, 'r') as f:
            obs = list(f.get('obs', {}).keys()) if 'obs' in f else []
            n = (f['obs']['_index'].shape[0] if 'obs' in f and '_index' in f['obs']
                 else None)
            has_ct = 'celltype' in obs
            has_l = 'leiden_res0_5' in obs
            print(p, 'n=', n, 'celltype=', has_ct, 'leiden_res0_5=', has_l,
                  'obs_keys=', obs[:25])
    except Exception as e:
        print(p, 'ERR', type(e).__name__, str(e)[:120])
"""


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    print("$", SCRIPT_FIND)
    _, o, e = c.exec_command(SCRIPT_FIND, timeout=300)
    out = o.read().decode(errors="replace")
    print(out)
    files = []
    for line in out.splitlines():
        parts = line.strip().split(" ", 2)
        if len(parts) == 3:
            files.append(parts[2])
    files = files[:25]
    quoted = " ".join(f"'{p}'" for p in files)
    cmd = f"HDF5_USE_FILE_LOCKING=FALSE {REMOTE_PY} - {quoted} <<'PY'\n{SCRIPT_PROBE}\nPY\n"
    print("$ probe", len(files), "files")
    _, o, e = c.exec_command(cmd, timeout=600)
    print(o.read().decode(errors="replace"))
    err = e.read().decode(errors="replace").rstrip()
    if err:
        print("[stderr]", err)
    c.close()


if __name__ == "__main__":
    main()
