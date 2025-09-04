"""最終更新日の更新を行うスクリプト"""

import subprocess


def lastupdate_refresh(
    path_list: list = ["./packages", "./develop", "./src", "./npy", "./docs", "./testpy", "./autostart_setup"]
):
    ps_command = "powershell -ExecutionPolicy RemoteSigned -File"
    ps_file = "./develop/UpdateTimeStamp.ps1"

    for path in path_list:
        cmd = f"{ps_command} {ps_file} {path}"
        print(cmd)
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    lastupdate_refresh()
