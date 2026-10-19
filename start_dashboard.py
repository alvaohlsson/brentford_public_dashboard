"""Start from any working directory: python /path/to/start_dashboard.py."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys

folder=Path(__file__).resolve().parent
if importlib.util.find_spec('streamlit') is None:
    print('Install the packages first using this command:')
    print(f'"{sys.executable}" -m pip install -r "{folder / "requirements.txt"}"')
    raise SystemExit(1)
os.chdir(folder)
raise SystemExit(subprocess.call([sys.executable,'-m','streamlit','run',str(folder/'app.py'),
                                 '--server.address=localhost','--browser.serverAddress=localhost',
                                 '--browser.gatherUsageStats=false']))
