import os
import subprocess
import itertools
import time
import csv
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---

# 1. Количество запусков для сбора статистической выборки
N_STAT_RUNS = 1000
N_WARMUP = 5

# 2. Имена входных файлов с результатами полного перебора
CBENCH_X86_RESULTS = "cbench_results_x86_64.csv"
CBENCH_ARM_RESULTS = "cbench_results_arm64.csv"
MIBENCH_X86_RESULTS = "mibench_results_x86_64.csv"
MIBENCH_ARM_RESULTS = "mibench_results_arm64.csv"

# 3. Конфигурации компилятора и бенчмарков
if platform.system() == "Darwin":
    COMPILER = "gcc-13"
else:
    COMPILER = "gcc"
BASE_OPT_LEVEL = "-O3"

# ... (Конфигурации CBENCH_CONFIG и MIBENCH_CONFIG скопированы сюда для полноты) ...
CBENCH_CONFIG = {
    "cat-cat1": {"type": "stdin"}, "cat-cat2": {"type": "stdin"}, "cat-cat3": {"type": "stdin"},
    "fac-fac1": {"type": "cpu"}, "fac-fac2": {"type": "cpu"}, "fac-fac3": {"type": "cpu"},
    "fac-fac4": {"type": "cpu"}, "fac-fac5": {"type": "argv", "args": ["1000"]},
    "fac-fac6": {"type": "argv", "args": ["1000"]}, "fac-facx": {"type": "cpu"}, "fac-facy": {"type": "cpu"},
    "qsort-qsort1": {"type": "cpu"}, "qsort-qsort2": {"type": "cpu"}, "qsort-qsort3": {"type": "cpu"},
    "qsort-qsort4": {"type": "cpu"}, "qsort-qsort5": {"type": "cpu"}, "sqrt-sqrt1": {"type": "cpu"},
    "sqrt-sqrt2": {"type": "cpu"}, "sqrt-sqrt3": {"type": "cpu"}, "malloc-malloc1": {"type": "cpu"},
    "malloc-malloc2": {"type": "cpu"}, "malloc-malloc3": {"type": "cpu"}, "malloc-malloc4": {"type": "cpu"},
}
MIBENCH_CONFIG = {
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

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def find_best_flags(results_x86_path, results_arm_path):
    """Находит лучшие комбинации флагов для каждой архитектуры из CSV файлов."""
    try:
        df_x86 = pd.read_csv(results_x86_path)
        df_arm = pd.read_csv(results_arm_path)
    except FileNotFoundError as e:
        print(f"ОШИБКА: Не найден файл с результатами: {e}. Запустите полный перебор сначала.")
        sys.exit(1)

    speedup_col = df_x86.columns[-1]
    flag_cols = df_x86.columns[1:-2].tolist()
    
    df_x86 = df_x86[df_x86[speedup_col] != "ERROR"].copy()
    df_x86[speedup_col] = pd.to_numeric(df_x86[speedup_col])
    df_arm = df_arm[df_arm[speedup_col] != "ERROR"].copy()
    df_arm[speedup_col] = pd.to_numeric(df_arm[speedup_col])

    benchmarks = sorted(list(set(df_x86['benchmark'].unique()) & set(df_arm['benchmark'].unique())))
    
    best_configs = {}
    for bench in benchmarks:
        best_x86_run = df_x86[df_x86['benchmark'] == bench].loc[df_x86[df_x86['benchmark'] == bench][speedup_col].idxmax()]
        best_arm_run = df_arm[df_arm['benchmark'] == bench].loc[df_arm[df_arm['benchmark'] == bench][speedup_col].idxmax()]
        best_configs[bench] = {
            "x86_64": best_x86_run[flag_cols].tolist(),
            "arm64": best_arm_run[flag_cols].tolist()
        }
    return best_configs

def run_and_collect_distribution(run_command_base, config, n_warmup, n_runs):
    """Запускает бенчмарк много раз и возвращает ВЕСЬ список времен."""
    run_command = list(run_command_base)
    system = platform.system()
    if system == "Linux":
        run_command.insert(0, "1"); run_command.insert(0, "-c"); run_command.insert(0, "taskset")
    elif system == "Darwin":
        # CRITICAL: We need an ABSOLUTE path to mac_taskset, because the subprocess
        # will change its current working directory (CWD).
        # We assume mac_taskset is in the same directory as this script.
        script_dir = Path(__file__).parent.resolve()
        mac_taskset_path = script_dir / "mac_taskset"

        if mac_taskset_path.exists():
             run_command.insert(0, "6"); run_command.insert(0, str(mac_taskset_path))
        else:
            # This is a critical error for benchmarking, so we should stop.
            print(f"\nОШИБКА: Утилита для привязки к ядру не найдена по пути: {mac_taskset_path}")
            print("Пожалуйста, скомпилируйте mac_taskset.c и положите его рядом со скриптом.")
            sys.exit(1)

    exec_times = []
    run_options = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE, "check": True}
    
    # Аргументы для cbench
    if "cbench" in str(config.get("root_dir", "")):
        cmd_with_args = run_command + config.get("args", [])
        run_type = config.get("type")
    # Аргументы для mibench
    else:
        cmd_with_args = list(run_command)
        if "args" in config:
            args = [str(config["root_dir"] / arg) if "input_data" in arg else arg for arg in config["args"]]
            cmd_with_args.extend(args)
        run_type = "argv" 

    # Прогрев
    for _ in tqdm(range(n_warmup), desc=f"Warmup ({run_type})"):
        if run_type == "stdin":
            with open(config["input_file"], "r") as f:
                subprocess.run(run_command, stdin=f, **run_options)
        else:
            subprocess.run(cmd_with_args, **run_options)
    
    # Замеры
    for i in tqdm(range(n_runs), desc=f"Measurement ({run_type})"):
        start_time = time.perf_counter()
        if run_type == "stdin":
            with open(config["input_file"], "r") as f:
                subprocess.run(run_command, stdin=f, **run_options)
        else:
            subprocess.run(cmd_with_args, **run_options)
        end_time = time.perf_counter()
        exec_times.append(end_time - start_time)
        
    return exec_times

def build_cbench(source_files, output_exe, flags):
    compile_command = [COMPILER, BASE_OPT_LEVEL] + flags + SYSTEM_SPECIFIC_CFLAGS.split() + source_files + ["-o", str(output_exe), "-lm", "-lgmp"]    
    subprocess.run(compile_command, check=True, capture_output=True, text=True)

def build_mibench(build_dir, make_target, flags):
    original_dir = Path.cwd()
    os.chdir(build_dir)
    try:
        subprocess.run(["make", "clean"], check=True, capture_output=True, text=True)
        env = os.environ.copy()
        env["CC"] = COMPILER
        env["CFLAGS"] = f"{BASE_OPT_LEVEL} {SYSTEM_SPECIFIC_CFLAGS} {' '.join(flags)} -lm"
        subprocess.run(["make", make_target], env=env, check=True, capture_output=True, text=True)
    finally:
        os.chdir(original_dir)

def run_suite(suite_name, root_dir, config_dict, results_csv, best_flags_map):
    """Запускает сбор стат. данных для целого набора (cbench или mibench)."""
    print(f"\n{'='*20} ЗАПУСК СБОРА СТАТИСТИКИ ДЛЯ: {suite_name.upper()} {'='*20}")
    
    if results_csv.exists(): results_csv.unlink()
    
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "flags_source_arch", "run_id", "exec_time"])

        for benchmark_name, champion_flags in best_flags_map.items():
            print(f"\n--- Тестирование бенчмарка: {benchmark_name} ---")
            
            ### КЛЮЧЕВАЯ ЛОГИКА: Итерируем по обоим наборам чемпионских флагов ###
            for arch_source, flags in champion_flags.items():
                print(f"  Тестирование чемпионских флагов от '{arch_source}'...", end="", flush=True)
                
                try:
                    # Логика сборки и запуска для cbench
                    if suite_name == "cbench":
                        source_files = [str(f) for f in (root_dir / "src").rglob(f"*/{benchmark_name.split('-')[1]}.c")]
                        output_exe = Path(f"bin/{benchmark_name}_stat")
                        build_cbench(source_files, output_exe, flags)
                        config = config_dict.get(benchmark_name, {})
                        config["root_dir"] = root_dir
                        config["input_file"] = root_dir.parent / "input_data/input.txt"
                        exec_times = run_and_collect_distribution([f"./{output_exe}"], config, N_WARMUP, N_STAT_RUNS)
                    
                    # Логика сборки и запуска для mibench
                    elif suite_name == "mibench":
                        config = config_dict.get(benchmark_name, {})
                        build_dir = root_dir / config["build_dir"]
                        make_target = config["make_target"]
                        output_exe = build_dir / config["run_executable"]
                        build_mibench(build_dir, make_target, flags)
                        config["root_dir"] = root_dir
                        exec_times = run_and_collect_distribution([str(output_exe)], config, N_WARMUP, N_STAT_RUNS)
                    
                    # Запись результатов
                    for i, exec_time in enumerate(exec_times):
                        writer.writerow([benchmark_name, arch_source, i + 1, exec_time])
                    print(f" Успех! Собрано {len(exec_times)} замеров.")

                except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
                    print(f" ОШИБКА!")
                    writer.writerow([benchmark_name, arch_source, "ERROR", str(e)])
                    with open(f"error_log_stat_{suite_name}.txt", "a") as log:
                        log.write(f"--- ОШИБКА ({benchmark_name}) для флагов от {arch_source} ---\n")
                        if isinstance(e, subprocess.CalledProcessError):
                            log.write(f"Stderr: {e.stderr}\n\n")
                        else:
                            log.write(f"Ошибка: {e}\n\n")

if __name__ == "__main__":
    # Определяем текущую архитектуру
    current_arch = platform.machine()
    if "arm" in current_arch or "aarch64" in current_arch:
        CPU_ARCH = "arm64"
    elif "x86_64" in current_arch:
        CPU_ARCH = "x86_64"
    else:
        print(f"Неизвестная архитектура: {current_arch}. Выход.")
        sys.exit(1)
        
    print(f"Обнаружена архитектура: {CPU_ARCH}")

    # --- ЗАПУСК ДЛЯ CBENCH ---
    print("\n--- Этап 1: Поиск лучших флагов для cBench ---")
    best_cbench_flags = find_best_flags(CBENCH_X86_RESULTS, CBENCH_ARM_RESULTS)
    if best_cbench_flags:
        cbench_stat_results_file = Path(f"cbench_stat_results_on_{CPU_ARCH}.csv")
        run_suite("cbench", Path("./cbench"), CBENCH_CONFIG, cbench_stat_results_file, best_cbench_flags)

    # --- ЗАПУСК ДЛЯ MIBENCH ---
    print("\n--- Этап 1: Поиск лучших флагов для MiBench ---")
    best_mibench_flags = find_best_flags(MIBENCH_X86_RESULTS, MIBENCH_ARM_RESULTS)
    if best_mibench_flags:
        mibench_stat_results_file = Path(f"mibench_stat_results_on_{CPU_ARCH}.csv")
        run_suite("mibench", Path("./mibench").resolve(), MIBENCH_CONFIG, mibench_stat_results_file, best_mibench_flags)

    print("\n--- Сбор данных для статистического анализа завершен! ---")
