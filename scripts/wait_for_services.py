import time
import socket


def wait(host, port):

    for _ in range(60):

        try:

            s = socket.create_connection((host, port), timeout=2)
            s.close()
            return

        except Exception:

            time.sleep(2)

    raise RuntimeError(f"{host}:{port} unavailable")


if __name__ == "__main__":
    wait("localhost", 5432)
    wait("localhost", 6379)
