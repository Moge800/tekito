import socket


def find_free_ports(start_port=1024, end_port=65535, max_results=10):
    free_ports = []
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1)  # タイムアウトを短く設定
            result = s.connect_ex(("127.0.0.1", port))
            if result != 0:  # 接続できなければ空きポート
                free_ports.append(port)
                if len(free_ports) >= max_results:
                    break
    return free_ports


if __name__ == "__main__":
    free_ports = find_free_ports()
    print("\n空きポート:", free_ports)
