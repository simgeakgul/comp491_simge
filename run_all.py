import subprocess
import sys
import  argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base",
        required=True,
        help="folder to process (will be passed to each script)"
    )
    return p.parse_args()

def main():

    args = parse_args()

    scripts = [
        # "generate_prompt.py",
        # "run_pano.py",
        # "run_fixes.py",
        "run_depth.py",
    ]

    for script in scripts:
        print(f"→ Running {script} with base={args.base}")
        try:
            subprocess.check_call([
                sys.executable,
                script,
                "--base", args.base
            ])
        except subprocess.CalledProcessError as e:
            print(f"✖ {script} failed (exit {e.returncode})")
            sys.exit(e.returncode)

    print("✔ All done.")

if __name__ == "__main__":
    main()
