import os
import sys
import subprocess
import time
import urllib.request

def is_backend_healthy(url="http://127.0.0.1:8000/health", timeout=2):
    try:
        req = urllib.request.urlopen(url, timeout=timeout)
        return req.getcode() == 200
    except Exception:
        return False

def ensure_backend_running():
    if is_backend_healthy():
        print("[OK] Backend FastAPI server is already running on http://127.0.0.1:8000.")
        return True

    print("PhaseGuard Launcher: Backend server is offline. Starting FastAPI backend...")
    python_exe = sys.executable
    cmd = [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]

    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)

    print("Waiting for backend startup (loading ML models & cloud DB sync)...")
    for i in range(45):
        time.sleep(1)
        if is_backend_healthy():
            print("\n[OK] Backend server is online and healthy!")
            return True
        print(".", end="", flush=True)

    print("\nWarning: Backend server check timed out. Proceeding to launch Streamlit anyway...")
    return False

def main():
    # Detect the Streamlit binary in the current active python environment
    python_dir = os.path.dirname(sys.executable)
    
    # On Windows, binaries are in the Scripts/ subdirectory
    streamlit_bin = os.path.join(python_dir, "Scripts", "streamlit.exe")
    
    # Fallback to general command if not found
    if not os.path.exists(streamlit_bin):
        streamlit_bin = os.path.join(python_dir, "streamlit")
        if not os.path.exists(streamlit_bin):
            streamlit_bin = "streamlit"
            
    print("==================================================")
    print("      PhaseGuard Unified Application Launcher     ")
    print("==================================================")
    
    # 1. Ensure Backend is Active
    ensure_backend_running()
    
    # 2. Determine target dashboard (default to Layer 2 streamlit_app/app.py or dashboard.py if arg supplied)
    target_app = "streamlit_app/app.py"
    if len(sys.argv) > 1:
        target_app = sys.argv[1]
    elif not os.path.exists(target_app) and os.path.exists("dashboard.py"):
        target_app = "dashboard.py"

    print(f"\nInitiating Streamlit dashboard ({target_app})...")
    print(f"Streamlit path: {streamlit_bin}")
    print(f"Python path: {sys.executable}\n")
    
    try:
        subprocess.run([streamlit_bin, "run", target_app])
    except KeyboardInterrupt:
        print("\nPhaseGuard Dashboard stopped by user.")
    except Exception as e:
        print(f"Error starting Streamlit: {e}")

if __name__ == "__main__":
    main()

