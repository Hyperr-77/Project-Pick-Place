import os
import subprocess
import sys

TARGET_SCRIPT = os.path.join(os.path.dirname(__file__), "example.py")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "flamegraph.svg")


def run_target_script():
    if not os.path.exists(TARGET_SCRIPT):
        raise FileNotFoundError(f"Script not found: {TARGET_SCRIPT}")

    print(f"Running {TARGET_SCRIPT} for profiling...")
    subprocess.run([sys.executable, TARGET_SCRIPT], check=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        run_target_script()
    else:
        print(f"Py-Spy wordt gestart om {OUTPUT_FILE} te genereren voor example.py...")

        cmd = [
            "py-spy",
            "record",
            "-o",
            OUTPUT_FILE,
            "--format",
            "flamegraph",
            "--",
            sys.executable,
            TARGET_SCRIPT,
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"\nSucces! De flame graph is opgeslagen als '{OUTPUT_FILE}'.")
            print("Open dit SVG-bestand in je browser (Chrome/Edge) om het te bekijken.")
        except FileNotFoundError:
            print("\nFout: py-spy is niet gevonden. Doe 'pip install py-spy' en controleer of het in je PATH staat.")
        except subprocess.CalledProcessError:
            print("\nFout bij het uitvoeren. Op sommige systemen vereist py-spy adminrechten.")
