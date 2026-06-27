"""Stream-download h5ad files from the xiyou wuxu host with md5 verification."""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"
PORT = 12264
USER = "wuxu"
PASSWORD = "Wuxu@96885"

TASKS = [
    {
        "remote": "/home/wuxu/jupyter/MJ/newh5ad/acc_new.h5ad",
        "local": Path("F:/cartifm/acc_new.h5ad"),
    },
    {
        "remote": "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad",
        "local": Path("F:/cartifm/outputs/EBR/EBR_new.h5ad"),
    },
]

CHUNK = 8 * 1024 * 1024  # 8 MiB


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def remote_md5(ssh: paramiko.SSHClient, remote: str) -> str:
    _, out, _ = ssh.exec_command(f"md5sum {remote}", timeout=600)
    line = out.read().decode().strip()
    return line.split()[0]


def download(sftp: paramiko.SFTPClient, remote: str, local: Path) -> str:
    local.parent.mkdir(parents=True, exist_ok=True)
    info = sftp.stat(remote)
    total = info.st_size
    print(f"  remote size: {fmt_bytes(total)} ({total} B)")
    h = hashlib.md5()
    f_remote = sftp.open(remote, "rb")
    f_remote.prefetch(total)
    start = time.time()
    last = start
    done = 0
    with open(local, "wb") as f_local:
        while True:
            buf = f_remote.read(CHUNK)
            if not buf:
                break
            f_local.write(buf)
            h.update(buf)
            done += len(buf)
            now = time.time()
            if now - last >= 5 or done == total:
                rate = done / max(1e-3, now - start)
                eta = (total - done) / max(1e-3, rate)
                pct = 100.0 * done / total
                print(
                    f"    {fmt_bytes(done)} / {fmt_bytes(total)} "
                    f"({pct:5.1f}%)  rate {fmt_bytes(int(rate))}/s  eta {eta:6.0f}s"
                )
                last = now
    f_remote.close()
    print(f"    finished in {time.time()-start:.1f}s")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["acc", "ebr"], help="download just one")
    args = ap.parse_args()

    selected = TASKS
    if args.only == "acc":
        selected = [TASKS[0]]
    elif args.only == "ebr":
        selected = [TASKS[1]]

    print("Connecting to", HOST)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                look_for_keys=False, allow_agent=False, timeout=30)
    sftp = ssh.open_sftp()
    sftp.get_channel().settimeout(7200)

    rc = 0
    for t in selected:
        remote = t["remote"]
        local = t["local"]
        print(f"--- {remote} -> {local}")
        rmd5 = remote_md5(ssh, remote)
        print(f"  remote md5: {rmd5}")
        lmd5 = download(sftp, remote, local)
        print(f"  local  md5: {lmd5}")
        if rmd5 != lmd5:
            print("  !! MD5 MISMATCH !!")
            rc = 2
        else:
            print("  md5 ok")

    sftp.close()
    ssh.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
