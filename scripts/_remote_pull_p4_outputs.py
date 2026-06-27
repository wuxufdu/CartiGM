"""Reconnect to the remote host and pull /home/wuxu/cartigsfm_remote/p4_outdir
TSVs + report into F:/cartifm/outputs/EBR_p4_remote.
"""
from __future__ import annotations

from pathlib import Path

import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"
PORT = 12264
USER = "wuxu"
PASSWORD = "Wuxu@96885"
REMOTE_BASE = "/home/wuxu/cartigsfm_remote/p4_outdir"
LOCAL_BASE = Path("F:/cartifm/outputs/EBR_p4_remote")

FILES = [
    "tsv/p4_self_sample_cluster_pseudobulk.tsv",
    "tsv/p4_self_sample_cluster_meta.tsv",
    "tsv/p4_sample_cluster_three_layer_scores.tsv",
    "tsv/p4_sample_cluster_top_assignments.tsv",
    "tsv/p4_tissue_axis_summary.tsv",
    "tsv/p4_marker_validation_table.tsv",
    "docs/P4_INDEPENDENT_VALIDATION_REPORT.md",
]


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    sftp = c.open_sftp()
    for rel in FILES:
        rp = f"{REMOTE_BASE}/{rel}"
        lp = LOCAL_BASE / rel
        lp.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(rp, str(lp))
        print("got", rel, "->", lp)
    sftp.close()
    c.close()


if __name__ == "__main__":
    main()
