"""
test_dashboard.py
-----------------
Smoke test for Streamlit dashboard in FairML.
"""

import os
import sys
import subprocess
import time

# Ensure src is on the Python path (in case dashboard imports from src)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)


def test_dashboard_runs():
    """Smoke test to ensure the Streamlit dashboard can start."""
    dashboard_path = os.path.join(
        os.path.dirname(__file__), "../dashboard/app.py"
    )
    success = False

    try:
        # Start the dashboard in headless mode
        proc = subprocess.Popen(
            ["streamlit", "run", dashboard_path, "--server.headless", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Give it a moment to start
        time.sleep(3)

        # If process is still running, assume it started successfully
        if proc.poll() is None:
            success = True

        # Terminate the process
        proc.terminate()
        proc.wait()

    except Exception as e:
        success = False
        print(f"Dashboard test failed with exception: {e}")

    assert success, "Streamlit dashboard failed to run"
