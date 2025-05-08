import subprocess
import sys

def run_julia():
    cmd = ["julia", "../trilat.jl"]  # calls root test_real_data.jl
    # Actually call the original script
    cmd = ["julia", "../../test_real_data.jl"]
    p = subprocess.run(cmd, cwd="/Users/saurav/Downloads/DSP/trilateration-main/present/python", capture_output=True, text=True)
    print("--- Julia Results ---")
    print(p.stdout)


def run_python():
    cmd = ["python3", "real_data.py"]
    p = subprocess.run(cmd, cwd="/Users/saurav/Downloads/DSP/trilateration-main/present/python", capture_output=True, text=True)
    print("--- Python Results ---")
    print(p.stdout)

if __name__ == '__main__':
    run_julia()
    run_python()
