import subprocess
import sys
import os
import webbrowser
import socket


def find_free_port(start=8501, max_tries=20):
    """Find the first available TCP port."""
    port = start
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
        port += 1
    raise RuntimeError("❌ No free port found!")


def main():
    # Path to streamlit UI
    ui_path = os.path.join(os.path.dirname(__file__), "ui.py")

    # Find free port automatically
    port = find_free_port(8501)

    # Print the URL
    url = f"http://localhost:{port}"
    print(f"\n🚀 Starting ControlProGen Web App on {url}\n")

    # Launch streamlit in subprocess
    p = subprocess.Popen([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        ui_path,
        "--server.port", str(port),
        "--server.headless", "true"
    ])

    # Open browser AFTER streamlit actually starts
    webbrowser.open(url)

    # Wait for process
    p.wait()