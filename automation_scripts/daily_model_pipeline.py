import subprocess
import sys
import os

def run_step(script_name):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.join(base_path, "scripts", script_name)

    print(f"\n--- Running Islamabad Daily Model Pipeline Step: {script_name} ---")
    result = subprocess.run([sys.executable, target], capture_output=False)

    if result.returncode != 0:
        print(f"[FAILED] Step {script_name} exited with error.")
        sys.exit(1)

if __name__ == "__main__":
    run_step("model_train.py")
    run_step("promote_model.py")
    print("\n[SUCCESS] Islamabad daily model retraining & promotion pipeline completed.")
