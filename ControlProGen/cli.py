import subprocess
import sys
import os
import webbrowser

def main():
    # Path to streamlit UI
    ui_path = os.path.join(os.path.dirname(__file__), "ui.py")

    # Automatically open browser
    webbrowser.open("http://localhost:8501")

    # Launch streamlit
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        ui_path
    ])