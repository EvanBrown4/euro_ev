import subprocess

def run_files():
    subprocess.run(["python", "rating_calc.py"])
    subprocess.run(["python", "cleaning.py"])
    subprocess.run(["python", "probability_builder.py"])

if __name__ == "__main__":
    run_files()