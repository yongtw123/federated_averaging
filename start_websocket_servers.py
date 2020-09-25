import subprocess
import sys
from pathlib import Path

python = Path(sys.executable).name

#FILE_PATH = Path(__file__).resolve().parents[4].joinpath("run_websocket_server.py")
FILE_PATH = Path(__file__).resolve().parent.joinpath("run_websocket_server.py")
FILE_PATH = str(FILE_PATH)

call_alice = [python, FILE_PATH, "--port", "8777", "--id", "alice"]
print("Starting server for Alice")
subprocess.Popen(call_alice)

call_bob = [python, FILE_PATH, "--port", "8778", "--id", "bob"]
print("Starting server for Bob")
subprocess.Popen(call_bob)

#call_charlie = [python, FILE_PATH, "--port", "8779", "--id", "charlie"]
#print("Starting server for Charlie")
#subprocess.Popen(call_charlie)


