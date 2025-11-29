import subprocess
import signal
import os
from typing import Optional, List


def run_experiment(
    config_file: str = "configs/TakeCover.gin",
    num_workers: int = 8,
    start_param_sync_service: bool = False,
    experiment_name: str = "test",
    bucket_name: str = "es_experiments",
    json_file: str = "gcs.json",
    extra_worker_args: Optional[List[str]] = None,
):
    """
    Runs the ES experiment workflow and stores logs into separate per-process files.
    Safe for Jupyter.
    """

    if extra_worker_args is None:
        extra_worker_args = []

    # Create log directory
    log_dir = os.path.join("logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    worker_processes = []
    worker_log_files = []

    # Options
    WORKER_OPTION = ["--master-address=127.0.0.1:10000"] if start_param_sync_service else []
    MASTER_OPTION = ["--start-param-sync-service"] if start_param_sync_service else []

    print(f"Starting {num_workers} workers...\n")

    # Launch workers
    for worker_id in range(num_workers):
        cmd = [
            "python", "run_worker.py",
            f"--config={config_file}",
            f"--worker-id={worker_id}",
        ] + WORKER_OPTION + extra_worker_args

        # Log file for this worker
        worker_log_path = os.path.join(log_dir, f"worker_{worker_id}.log")
        worker_log = open(worker_log_path, "w")

        p = subprocess.Popen(cmd, stdout=worker_log, stderr=worker_log)
        worker_processes.append(p)
        worker_log_files.append(worker_log)

        print(f"Worker {worker_id} started (PID={p.pid}) → {worker_log_path}")

    print("\nStarting master...\n")

    # Launch master
    master_cmd = [
        "python", "run_master.py",
        f"--config={config_file}",
        f"--num-workers={num_workers}",
        f"--gcs-bucket={bucket_name}",
        f"--gcs-experiment-name={experiment_name}",
        f"--gcs-credential={json_file}",
    ] + MASTER_OPTION

    master_log_path = os.path.join(log_dir, "master.log")
    master_log = open(master_log_path, "w")

    master_proc = subprocess.Popen(master_cmd, stdout=master_log, stderr=master_log)
    print(f"Master started (PID={master_proc.pid}) → {master_log_path}")

    master_proc.wait()
    master_log.close()

    print("\nMaster finished. Terminating workers...\n")

    # Kill workers
    for p, log_file in zip(worker_processes, worker_log_files):
        try:
            p.send_signal(signal.SIGTERM)
            print(f"Killed worker PID={p.pid}")
        except Exception as e:
            print(f"Could not kill worker PID={p.pid}: {e}")
        finally:
            log_file.close()

    print("\nAll workers terminated.")
    print(f"Logs saved under: {log_dir}")
    print("Done.")
