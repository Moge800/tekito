import http.client


def try_http_connection(host="localhost", port=8080):
    try:
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        print(f"HTTP {response.status} {response.reason}")
        print(response.read().decode())
        conn.close()
    except Exception as e:
        print(f"接続エラー: {e}")


try_http_connection()
