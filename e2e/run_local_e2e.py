"""Run the Selenium smoke test locally (no Docker).

Starts the backend (port 8010, throw-away database) and the already-built frontend
(`vite preview`, port 4173), runs e2e/test_feedback_flow.py with the Chrome installed
on this machine, and always shuts both servers down again.

Needs: `npm run build` already done in frontend/, Chrome installed, `pip install -r e2e/requirements.txt`.
Optional: CHROME_BIN=<path to chrome executable>.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_PORT, FRONTEND_PORT = 8010, 4173


def wait_for(url: str, timeout: float = 60) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3):
                return True
        except OSError:
            time.sleep(1)
    return False


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    node = shutil.which("node")
    vite = ROOT / "frontend" / "node_modules" / "vite" / "bin" / "vite.js"
    if not node or not vite.exists() or not (ROOT / "frontend" / "dist").is_dir():
        print("ERROR: run `npm ci && npm run build` in frontend/ first (and make sure node is on PATH)")
        return 1
    api = f"http://127.0.0.1:{BACKEND_PORT}"
    site = f"http://127.0.0.1:{FRONTEND_PORT}"
    processes = []
    with tempfile.TemporaryDirectory(prefix="feedback-e2e-") as tmp:
        tmp_path = Path(tmp)
        backend_env = {**os.environ, "SMART_FEEDBACK_DB": str(tmp_path / "e2e.db"), "SMART_FEEDBACK_MODELS": str(tmp_path / "models")}
        frontend_env = {**os.environ, "VITE_API_TARGET": api}
        logs = {name: open(tmp_path / f"{name}.log", "w") for name in ("backend", "frontend")}
        try:
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
                cwd=ROOT / "backend", env=backend_env, stdout=logs["backend"], stderr=subprocess.STDOUT))
            processes.append(subprocess.Popen(
                [node, str(vite), "preview", "--host", "127.0.0.1", "--port", str(FRONTEND_PORT), "--strictPort"],
                cwd=ROOT / "frontend", env=frontend_env, stdout=logs["frontend"], stderr=subprocess.STDOUT))
            if not wait_for(f"{api}/api/health") or not wait_for(site):
                print("ERROR: servers did not start in time")
                code = 1
            else:
                test_env = {**os.environ, "APP_URL": site, "SELENIUM_URL": ""}
                code = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", str(ROOT / "e2e" / "test_feedback_flow.py")],
                    cwd=ROOT / "e2e", env=test_env).returncode
            if code != 0:
                for name, handle in logs.items():
                    handle.flush()
                    print(f"--- {name} log (tail) ---")
                    print("".join((tmp_path / f"{name}.log").read_text(errors="replace").splitlines(True)[-20:]))
            return code
        finally:
            for process in processes:
                stop(process)
            for handle in logs.values():
                handle.close()


if __name__ == "__main__":
    sys.exit(main())
