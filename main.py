import subprocess
import time
import sys

# Define the processes to run
processes = [
    ["python", "laptop_server/server.py"],   # WebSocket Bridge
    ["python", "backend/server.py"],        # Data Processing & AI
    ["streamlit", "run", "dashboard/app.py"] # UI Layer
]

running_processes = []

print("🚀 Starting SyncFlow AI System...")

try:
    for cmd in processes:
        p = subprocess.Popen(cmd)
        running_processes.append(p)
        time.sleep(2) # Give each process a moment to bind to its port

    print("\n✅ All systems active. Press Ctrl+C to stop everything.")
    
    # Keep the main script alive while sub-processes are running
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nTerminating all SyncFlow services...")
    for p in running_processes:
        p.terminate()
    print("👋 System shutdown complete.")
    sys.exit()