from __future__ import annotations
import paramiko

HOST = "sw2-dynamic.xiyoucloud.pro"; PORT = 12264
USER = "wuxu"; PASSWORD = "Wuxu@96885"


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
              timeout=30, look_for_keys=False, allow_agent=False)
    cmd = (
        "ls -la /home/wuxu/cartigsfm_remote/cartigsfm/resources/dictionary_v1/cartilage_dictionary_v1.json; "
        "/home/wuxu/miniconda3/envs/squidp311/bin/python -c \""
        "import json; "
        "d=json.load(open('/home/wuxu/cartigsfm_remote/cartigsfm/resources/dictionary_v1/cartilage_dictionary_v1.json'));"
        "print('version', d.get('version'));"
        "ax=[a for a in d['layers']['cell_subtype']['axes'] if a['axis_id']=='cell_subtype::Effector_Metabolic_Chondrocytes'][0];"
        "print('anti_n', len(ax.get('anti_genes',[])));"
        "print('COL2A1 in anti weights:', 'COL2A1' in ax.get('anti_marker_weights',{}));"
        "print('FMOD weight:', ax.get('anti_marker_weights',{}).get('FMOD'));"
        "\""
    )
    _, o, e = c.exec_command(cmd, timeout=30)
    print(o.read().decode())
    err = e.read().decode().rstrip()
    if err:
        print('[stderr]', err)
    c.close()

if __name__ == "__main__":
    main()
