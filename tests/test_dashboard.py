"""
test_dashboard.py
-----------------
Smoke test for Streamlit dashboard in FairML.
"""

import os
import subprocess
import time


def test_dashboard_runs():
    """Smoke test to ensure the Streamlit dashboard can start."""
    dashboard_path = os.path.join(
        os.path.dirname(__file__), "../dashboard/app.py"
    )
    success = False

    try:
        proc = subprocess.Popen(
            ["streamlit", "run", dashboard_path, "--server.headless", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        time.sleep(3)

        if proc.poll() is None:
            success = True

        proc.terminate()
        proc.wait()

    except Exception as e:
        success = False
        print(f"Dashboard test failed with exception: {e}")

    assert success, "Streamlit dashboard failed to run"
