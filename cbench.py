import os
import subprocess
import itertools
import time
import csv
from pathlib import Path
import platform
import numpy as np

# --- Experiment configuration ---

# 1. Directories and files
BENCHMARK_ROOT_DIR = Path("./cbench/src")
INPUT_DATA_DIR = Path("./input_data")

CPU_ARCH = 'x86_64'

RESULTS_CSV = f"cbench_results_{CPU_ARCH}.csv"
ERROR_LOG = "error_log.txt"

# 2. Configuration for I/O-bound benchmarks
INPUT_FILENAME = "input.txt"
INPUT_FILE_SIZE_MB = 10

# 3. Compiler
#    On Linux this is usually "gcc".
#    On macOS after installing via Homebrew, this is usually "gcc-13" or "gcc-14".
COMPILER = "gcc" 

# 4. Baseline optimization level
BASE_OPT_LEVEL = "-O3"

# 5. Number of runs for averaging
N_EXECUTE = 30
N_WARMUP = 2

# 6. Flags for testing
FLAGS_TO_TEST = {
    "tree-vectorize": ["-ftree-vectorize", "-fno-tree-vectorize"],
    "unroll-loops": ["-funroll-loops", "-fno-unroll-loops"],
    "slp-vectorize": ["-ftree-slp-vectorize", "-fno-tree-slp-vectorize"],
    "inline-functions": ["-finline-functions", "-fno-inline-functions"],
    "prefetch-loop-arrays": ["-fprefetch-loop-arrays", "-fno-prefetch-loop-arrays"],
    "omit-frame-pointer": ["-fomit-frame-pointer", "-fno-omit-frame-pointer"],
    "align-functions": ["-falign-functions=1", "-falign-functions=16", "-falign-functions=32", "-falign-functions=64"],
}

# 7. Types of runs for each benchmark
#    'cpu':   Run without arguments and without stdin (for CPU-bound tasks)
#    'stdin': Run with redirection of stdin from a file
#    'argv':  Run with passing of command-line arguments
BENCHMARK_CONFIG = {
    # --- cat ---
    "cat-cat1":    {"type": "stdin"},
    "cat-cat2":    {"type": "stdin"},
    "cat-cat3":    {"type": "stdin"},

    # --- fac ---
    "fac-fac1":    {"type": "cpu"},
    "fac-fac2":    {"type": "cpu"},
    "fac-fac3":    {"type": "cpu"},
    "fac-fac4":    {"type": "cpu"},
    "fac-fac5":    {"type": "argv", "args": ["1000"]},
    "fac-fac6":    {"type": "argv", "args": ["1000"]},
    "fac-facx":    {"type": "cpu"},
    "fac-facy":    {"type": "cpu"},

    # --- qsort, sqrt, malloc ---
    "qsort-qsort1": {"type": "cpu"},
    "qsort-qsort2": {"type": "cpu"},
    "qsort-qsort3": {"type": "cpu"},
    "qsort-qsort4": {"type": "cpu"},
    "qsort-qsort5": {"type": "cpu"},
    "sqrt-sqrt1":   {"type": "cpu"},
    "sqrt-sqrt2":   {"type": "cpu"},
    "sqrt-sqrt3":   {"type": "cpu"},
    "malloc-malloc1": {"type": "cpu"},
    "malloc-malloc2": {"type": "cpu"},
    "malloc-malloc3": {"type": "cpu"},
    "malloc-malloc4": {"type": "cpu"},
}

# --- CONFIG END ---


def create_input_file_if_not_exists():
    INPUT_DATA_DIR.mkdir(exist_ok=True)
    input_path = INPUT_DATA_DIR / INPUT_FILENAME
    if not input_path.is_file():
        print(f"Creating test file {input_path} of size {INPUT_FILE_SIZE_MB}MB...")
        base_string = "The quick brown fox jumps over the lazy dog. 1234567890. " * 20 + "\n"
        num_writes = (INPUT_FILE_SIZE_MB * 1024 * 1024) // len(base_string.encode('utf-8'))
        with open(input_path, "w") as f:
            for _ in range(num_writes):
                f.write(base_string)
        print("Test file created.")
    return input_path

def discover_benchmarks(root_dir):
    benchmarks = {}
    for c_file in sorted(root_dir.rglob('*.c')):
        benchmark_name = f"{c_file.parent.name}-{c_file.stem}"
        benchmarks[benchmark_name] = [str(c_file)]
    return benchmarks

def generate_flag_combinations(flags_dict):
    flag_names = list(flags_dict.keys())
    flag_options = list(flags_dict.values())
    combinations = list(itertools.product(*flag_options))
    return flag_names, combinations

def run_and_measure(benchmark_name, run_command_base, run_type, input_file_path, n_warmup, n_runs):
    """Helper function for multiple runs and time measurement."""
    run_command = list(run_command_base)
    system = platform.system()
    if system == "Linux":
        run_command.insert(0, "1")
        run_command.insert(0, "-c")
        run_command.insert(0, "taskset")
    elif system == "Darwin":
        run_command.insert(0, "-i")
        run_command.insert(0, "-B")
        run_command.insert(0, "taskpolicy")
    
    exec_times = []
    run_options = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE, "check": True}

    cmd_with_args = run_command + BENCHMARK_CONFIG.get(benchmark_name, {}).get("args", [])

    # Warmup runs
    for _ in range(n_warmup):
        if run_type == "stdin":
            with open(input_file_path, "r") as stdin_file:
                run_options["stdin"] = stdin_file
                subprocess.run(run_command, **run_options)
        else:
            subprocess.run(cmd_with_args, **run_options)

    # Measurements
    for _ in range(n_runs):
        start_time = time.perf_counter()
        if run_type == "stdin":
            with open(input_file_path, "r") as stdin_file:
                run_options["stdin"] = stdin_file
                subprocess.run(run_command, **run_options)
        else:
            subprocess.run(cmd_with_args, **run_options)
        end_time = time.perf_counter()
        exec_times.append(end_time - start_time)
        
    if not exec_times:
        raise ValueError("Failed to get any measurements.")
    
    return np.median(exec_times)

def main():
    """Main function for the experiment."""
    if os.path.exists(RESULTS_CSV):
        print(f"Deleting old results file: {RESULTS_CSV}")
        os.remove(RESULTS_CSV)
    if os.path.exists(ERROR_LOG):
        os.remove(ERROR_LOG)
    
    input_file_path = create_input_file_if_not_exists()
    benchmarks = discover_benchmarks(BENCHMARK_ROOT_DIR)
    
    flag_names, combinations = generate_flag_combinations(FLAGS_TO_TEST)
    total_runs = len(combinations) * len(benchmarks)

    print("--- Experiment start ---")
    print(f"Compiler: {COMPILER}")
    print(f"Number of runs on configuration: {N_EXECUTE} (+{N_WARMUP} warmup runs)")
    
    system = platform.system()
    if system == "Linux":
        print("Detected Linux: will use 'taskset -c 1' to bind to CPU core.")
    elif system == "Darwin":
        print("Detected macOS: will use 'taskpolicy -B -i' to increase priority.")
    
    print(f"Benchmarks found: {len(benchmarks)}")
    print(f"Total number of configurations for testing: {total_runs}")
    print("--------------------------------------------------------")

    Path("bin").mkdir(exist_ok=True)
    
    with open(RESULTS_CSV, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["benchmark"] + flag_names + ["median_exec_time_sec", "speedup_vs_O3"])

        run_counter, successful_runs, failed_runs = 0, 0, 0

        for benchmark_name, source_files in benchmarks.items():
            print(f"\n--- Testing benchmark: {benchmark_name} ---")
            
            baseline_time = 0.0
            baseline_exe = Path(f"bin/{benchmark_name}_baseline")
            try:
                print("  Building and measuring baseline (-O3)...", end="", flush=True)
                baseline_compile_command = [COMPILER, BASE_OPT_LEVEL] + source_files + ["-o", str(baseline_exe), "-lm", "-lgmp"]
                subprocess.run(baseline_compile_command, check=True, capture_output=True, text=True)
                
                config = BENCHMARK_CONFIG.get(benchmark_name, {"type": "cpu"})
                baseline_time = run_and_measure(benchmark_name, [f"./{baseline_exe}"], config["type"], input_file_path, N_WARMUP, N_EXECUTE)
                print(f" Success! Baseline time: {baseline_time:.6f} sec.")
            except (subprocess.CalledProcessError, ValueError) as e:
                print(" ERROR: Baseline run! Skipping benchmark.")
                with open(ERROR_LOG, "a") as log:
                    log.write(f"--- ERROR: Baseline run ({benchmark_name}) ---\n")
                    if isinstance(e, subprocess.CalledProcessError):
                        log.write(f"Command: {' '.join(map(str, e.cmd))}\n")
                        log.write(f"Stderr: {e.stderr}\n\n")
                    else:
                        log.write(f"Error: {e}\n\n")
                if baseline_exe.exists():
                    os.remove(baseline_exe)
                continue
            finally:
                if baseline_exe.exists():
                    os.remove(baseline_exe)

            for i, flag_combo in enumerate(combinations):
                run_counter += 1
                print(f"  [Configuration {run_counter}/{total_runs}] Combination {i+1}/{len(combinations)}...", end="", flush=True)

                output_exe = Path(f"bin/{benchmark_name}_{i}")
                compile_command = [COMPILER, BASE_OPT_LEVEL] + list(flag_combo) + source_files + ["-o", str(output_exe), "-lm", "-lgmp"]

                try:
                    subprocess.run(compile_command, check=True, capture_output=True, text=True)
                    
                    config = BENCHMARK_CONFIG.get(benchmark_name, {"type": "cpu"})
                    median_exec_time = run_and_measure(benchmark_name, [f"./{output_exe}"], config["type"], input_file_path, N_WARMUP, N_EXECUTE)
                    
                    speedup = baseline_time / median_exec_time if median_exec_time > 0 else 0
                    writer.writerow([benchmark_name] + list(flag_combo) + [median_exec_time, speedup])
                    
                    successful_runs += 1
                    print(f" Success! Median execution time: {median_exec_time:.6f} sec. Speedup: {speedup:.4f}x")

                except (subprocess.CalledProcessError, ValueError) as e:
                    failed_runs += 1
                    print(f" ERROR!")
                    writer.writerow([benchmark_name] + list(flag_combo) + ["ERROR", "ERROR"])
                    with open(ERROR_LOG, "a") as log:
                        log.write(f"--- RUN ERROR {run_counter} ({benchmark_name}) ---\n")
                        if isinstance(e, subprocess.CalledProcessError):
                            log.write(f"Command: {' '.join(map(str, e.cmd))}\n")
                            log.write(f"Stderr: {e.stderr}\n\n")
                        else:
                            log.write(f"Error: {e}\n\n")
                
                finally:
                    csv_file.flush()
                    if output_exe.exists():
                        os.remove(output_exe)

    print("\n--- Experiment finished! ---")
    print(f"Results saved to file: {RESULTS_CSV}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs} (details in {ERROR_LOG})")

if __name__ == "__main__":
    main()
