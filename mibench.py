import os
import subprocess
import itertools
import time
import csv
from pathlib import Path
import platform
import numpy as np
import sys

# --- CONFIG ---

# 1. Dirs
CPU_ARCH = 'arm64'

MIBENCH_ROOT_DIR = Path("./mibench")
RESULTS_CSV = Path(f"mibench_results_{CPU_ARCH}.csv")
ERROR_LOG = Path("mibench_subset_error_log.txt")

# 2. Compiler and flags
# COMPILER = "gcc"
if platform.system() == "Darwin":
    COMPILER = "gcc-13"
else:
    COMPILER = "gcc"

BASE_OPT_LEVEL = "-O3"
N_EXECUTE = 30
N_WARMUP = 2

FLAGS_TO_TEST = {
    "tree-vectorize": ["-ftree-vectorize", "-fno-tree-vectorize"],
    "unroll-loops": ["-funroll-loops", "-fno-unroll-loops"],
    "slp-vectorize": ["-ftree-slp-vectorize", "-fno-tree-slp-vectorize"],
    "inline-functions": ["-finline-functions", "-fno-inline-functions"],
    "prefetch-loop-arrays": ["-fprefetch-loop-arrays", "-fno-prefetch-loop-arrays"],
    "omit-frame-pointer": ["-fomit-frame-pointer", "-fno-omit-frame-pointer"],
    "align-functions": ["-falign-functions=1", "-falign-functions=16", "-falign-functions=32", "-falign-functions=64"],
}

BENCHMARK_CONFIG = {
    "automotive_basicmath": {
        "build_dir": "automotive/basicmath", "make_target": "basicmath", "run_executable": "basicmath",
        "input_type": "argv", "args": ["10"]
    },
    "automotive_bitcount": {
        "build_dir": "automotive/bitcount", "make_target": "bitcnts", "run_executable": "bitcnts",
        "input_type": "argv", "args": ["50000000"]
    },
    "automotive_susan_s": {
        "build_dir": "automotive/susan", "make_target": "susan", "run_executable": "susan",
        "input_type": "script", "input_gen": [sys.executable, "input_generation/generate_susan_input.py", "1920", "1080"], 
        "args": ["input_data/susan_input.pgm", "output_susan.pgm", "-s"]
    },
    "network_dijkstra": {
        "build_dir": "network/dijkstra", "make_target": "dijkstra", "run_executable": "dijkstra",
        "input_type": "script", "input_gen": [sys.executable, "input_generation/generate_dijkstra_input.py", "10000"],
        "args": ["200", "input_data/dijkstra_input.dat"] 
    },
}

SYSTEM_SPECIFIC_CFLAGS = ""
if platform.system() == "Darwin":
    HOMEBREW_PREFIX = "/opt/homebrew"
    SYSTEM_SPECIFIC_CFLAGS = f"-I{HOMEBREW_PREFIX}/include -L{HOMEBREW_PREFIX}/lib"


# --- CONFIG END ---

def prepare_inputs():
    print("--- input data preparation ---")
    (MIBENCH_ROOT_DIR / "input_data").mkdir(exist_ok=True)
    original_dir = Path.cwd()
    os.chdir(MIBENCH_ROOT_DIR)
    unique_gens = {tuple(config["input_gen"]) for config in BENCHMARK_CONFIG.values() if config.get("input_type") == "script"}
    for gen_command in unique_gens:
        print(f"  Generation with: {' '.join(gen_command)}...")
        try:
            subprocess.run(gen_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"    WARNING: failed to generate data: {e.stderr}")
    os.chdir(original_dir)
    print("--- input data ready ---\n")

def build_benchmark(build_dir, make_target, cflags):
    """Builds one specific benchmark with given flags."""
    original_dir = Path.cwd()
    os.chdir(build_dir)
    try:
        subprocess.run(["make", "clean"], check=True, capture_output=True, text=True)
        env = os.environ.copy()
        env["CC"] = COMPILER
        env["CFLAGS"] = f"{cflags} {SYSTEM_SPECIFIC_CFLAGS}"
        subprocess.run(["make", make_target], env=env, check=True, capture_output=True, text=True)
    finally:
        os.chdir(original_dir)

def run_and_measure(executable_path, config, n_warmup, n_runs):
    """Runs one benchmark N times and returns the median time."""
    
    base_command = [str(executable_path)]
    system = platform.system()
    if system == "Linux":
        base_command.insert(0, "1")
        base_command.insert(0, "-c")
        base_command.insert(0, "taskset")
    elif system == "Darwin":
        # Using a custom utility (`mac_taskset`) that uses the Apple-supported
        # `thread_policy_set` API, which is not blocked by System Integrity Protection (SIP).
        # We will use the physical core number `6` (a P-Core) as the "affinity tag".

        # CRITICAL: We need an ABSOLUTE path to mac_taskset, because the subprocess
        # will change its current working directory (CWD).
        mac_taskset_path = Path.cwd().resolve() / "mac_taskset"
        
        base_command.insert(0, "6")                      # Affinity tag
        base_command.insert(0, str(mac_taskset_path))    # Absolute path to our utility


    if "args" in config:
        args = [str(MIBENCH_ROOT_DIR / arg) if "input_data" in arg else arg for arg in config["args"]]
        base_command.extend(args)

    run_env = os.environ.copy()
    if "env_vars" in config:
        run_env.update(config["env_vars"])

    # Warmup runs
    for _ in range(n_warmup):
        subprocess.run(base_command, cwd=executable_path.parent, env=run_env, check=True, capture_output=True, text=True)

    # Measurements
    exec_times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        subprocess.run(base_command, cwd=executable_path.parent, env=run_env, check=True, capture_output=True, text=True)
        end_time = time.perf_counter()
        exec_times.append(end_time - start_time)
    
    if not exec_times:
        raise ValueError("Failed to get any measurements.")
    
    return np.median(exec_times)

def generate_flag_combinations(flags_dict):
    """Generates all possible flag combinations."""
    flag_names = list(flags_dict.keys())
    flag_options = list(flags_dict.values())
    combinations = list(itertools.product(*flag_options))
    return flag_names, combinations

def main():
    global MIBENCH_ROOT_DIR
    MIBENCH_ROOT_DIR = MIBENCH_ROOT_DIR.resolve()
    
    if RESULTS_CSV.exists(): RESULTS_CSV.unlink()
    if ERROR_LOG.exists(): ERROR_LOG.unlink()

    prepare_inputs()
    flag_names, combinations = generate_flag_combinations(FLAGS_TO_TEST)
    
    print("--- Experiment start ---")
    print(f"Compiler: {COMPILER}")
    print(f"Number of runs on configuration: {N_EXECUTE} (+{N_WARMUP} warmup runs)")
    
    system = platform.system()
    if system == "Linux":
        print("Detected Linux: will use 'taskset -c 1' to bind to CPU core.")
    elif system == "Darwin":
        print("Detected macOS: will use 'mac_taskset' to pin to a Performance Core.")
    
    print(f"Benchmarks for testing: {', '.join(BENCHMARK_CONFIG.keys())}")
    print(f"Total number of flag combinations: {len(combinations)}")
    print("--------------------------------------------------------")

    with open(RESULTS_CSV, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["benchmark"] + flag_names + ["median_exec_time_sec", "speedup_vs_O3"])

        for name, config in BENCHMARK_CONFIG.items():
            print(f"\n--- Testing benchmark: {name} ---")
            build_dir = MIBENCH_ROOT_DIR / config["build_dir"]
            make_target = config["make_target"]
            run_executable = config["run_executable"]
            executable_path = build_dir / run_executable

            baseline_time = 0.0
            try:
                print("  Building and measuring baseline (-O3)...", end="", flush=True)
                build_benchmark(build_dir, make_target, BASE_OPT_LEVEL)
                baseline_time = run_and_measure(executable_path, config, N_WARMUP, N_EXECUTE)
                print(f" Success! Baseline time: {baseline_time:.6f} sec.")
            except (subprocess.CalledProcessError, ValueError) as e:
                print(" ERROR: Baseline run! Skipping benchmark.")
                with open(ERROR_LOG, "a") as log:
                    log.write(f"--- ERROR: Baseline run ({name}) ---\n")
                    if isinstance(e, subprocess.CalledProcessError):
                        log.write(f"Stderr: {e.stderr}\n\n")
                    else:
                        log.write(f"Error: {e}\n\n")
                continue

            for i, flag_combo in enumerate(combinations):
                cflags = f"{BASE_OPT_LEVEL} {SYSTEM_SPECIFIC_CFLAGS} {' '.join(flag_combo)}"
                print(f"  [Configuration {i+1}/{len(combinations)}]...", end="", flush=True)

                try:
                    build_benchmark(build_dir, make_target, cflags)
                    median_time = run_and_measure(executable_path, config, N_WARMUP, N_EXECUTE)
                    speedup = baseline_time / median_time if median_time > 0 else 0
                    
                    writer.writerow([name] + list(flag_combo) + [median_time, speedup])
                    print(f" Success! Median execution time: {median_time:.6f} sec. Speedup: {speedup:.4f}x")

                except (subprocess.CalledProcessError, ValueError) as e:
                    print(f" ERROR!")
                    writer.writerow([name] + list(flag_combo) + ["ERROR", "ERROR"])
                    with open(ERROR_LOG, "a") as log:
                        log.write(f"--- ERROR ({name}) for CFLAGS='{cflags}' ---\n")
                        if isinstance(e, subprocess.CalledProcessError):
                            log.write(f"Stderr: {e.stderr}\n\n")
                        else:
                            log.write(f"Error: {e}\n\n")
                finally:
                    csv_file.flush()

    print("\n--- Experiment finished! ---")
    print(f"Results saved to file: {RESULTS_CSV}")

if __name__ == "__main__":
    main()
