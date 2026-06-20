"""Connect to the remote host and dump EBR.h5ad's obs schema for P4 setup."""
from __future__ import annotations

import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"
PORT = 12264
USER = "wuxu"
PASSWORD = "Wuxu@96885"

REMOTE_H5 = "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad"

PROBE = r'''
import sys
try:
    import anndata as ad
except Exception as e:
    print("NO_ANNDATA:", e); sys.exit(0)
import h5py
# Try backed first; fall back to opening just obs/uns shape via h5py
try:
    a = ad.read_h5ad("/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad", backed="r")
    backed_ok = True
except Exception as e:
    print("BACKED_FAILED:", repr(e))
    backed_ok = False
if not backed_ok:
    f = h5py.File("/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad", "r")
    print("h5py keys:", list(f.keys()))
    print("X shape:", f["X"].attrs.get("shape", None) or (f["X/indptr"].shape if "X/indptr" in f else None))
    print("obs cols (h5):", list(f["obs"].keys()))
    obs = f["obs"]
    for c in list(obs.keys()):
        ds = obs[c]
        if isinstance(ds, h5py.Group) and "categories" in ds:
            cats = [x.decode() if isinstance(x, bytes) else str(x) for x in ds["categories"][:60]]
            codes = ds["codes"][...]
            import numpy as _np
            uniq, cnt = _np.unique(codes, return_counts=True)
            order = (-cnt).argsort()
            top = {cats[uniq[i]] if uniq[i] < len(cats) else "?": int(cnt[i]) for i in order[:20]}
            print("CAT", c, "n", len(cats), "top", top)
        else:
            try:
                arr = ds[:5]
                print("RAW", c, "head", list(arr.tolist()) if hasattr(arr,'tolist') else list(arr))
            except Exception as ee:
                print("RAW", c, "err", ee)
    sys.exit(0)
print("shape", a.shape)
print("obs_columns", list(a.obs.columns))
print("layers", list(a.layers.keys()))
print("uns_keys", list(a.uns.keys())[:10])
print("obsm_keys", list(a.obsm.keys()))
for c in a.obs.columns:
    s = a.obs[c]
    try:
        nu = int(s.astype(str).nunique())
    except Exception:
        nu = -1
    head = list(s.astype(str).iloc[:5])
    if nu < 60 and nu >= 0:
        vc = s.astype(str).value_counts().head(20)
        print("COL", c, "nunique", nu, "top", vc.to_dict())
    else:
        print("COL", c, "nunique", nu, "head", head)
'''


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 600) -> None:
    print("$", cmd)
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors="replace").rstrip()
    err = e.read().decode(errors="replace").rstrip()
    if out:
        print(out)
    if err:
        print("[stderr]", err)


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    # Probe each conda env for a working anndata >= 0.10
    probe_one = (
        'for env in $(ls /home/wuxu/miniconda3/envs/); do '
        '  py=/home/wuxu/miniconda3/envs/$env/bin/python; '
        '  test -x "$py" || continue; '
        '  v=$($py -c "import anndata, sys; print(anndata.__version__)" 2>/dev/null); '
        '  echo "ENV $env -> $v"; '
        'done'
    )
    run(c, probe_one, timeout=120)
    py = None
    sftp = c.open_sftp()
    script_path = "/tmp/_probe_ebr_obs.py"
    with sftp.open(script_path, "w") as f:
        f.write(PROBE)
    sftp.close()
    # Try the most likely envs in order
    for env in ("omicverse", "rapids_singlecell", "scib", "squidp311"):
        cand = f"/home/wuxu/miniconda3/envs/{env}/bin/python"
        run(c, f"{cand} -c 'import anndata; print(anndata.__version__)' 2>&1 | head -3")
        run(c, f"{cand} {script_path} 2>&1 | tail -200", timeout=1800)
        print("---")
    c.close()


if __name__ == "__main__":
    main()
