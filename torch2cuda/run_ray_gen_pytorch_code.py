'''
Usage:
python run_ray_gen_pytorch_code.py \
    --output_file <output_jsonl_fn> \
    --level <level_int> \
    --target_count <target_count_int>

Generate PyTorch code samples. Please refer to generator/gen.py on how to generate samples.

Basically, this script is logically equivalent to:
python generator/gen.py \
    --output_file <output_jsonl_fn> \
    --level <level_int> \
    --target_count <target_count_int>
    
Except that, this script use ray to distribute workloads. Refer to run_ray_rollout.py for the usage of ray.

In the main ray process, it creates workers and generate jobs to job input queue. It listen to job output queue, find any finished results and dump it to <output_jsonl_fn> immediately.
It can resume from previously interruptted output_jsonl_fn.

In each worker, it needs 1 GPUs and 1 CPU. It reads jobs from the job input queue, generate a sample, and write sample to the job output queue.
'''

import argparse
import json
import logging
import os
import signal
import time
from typing import Dict, cast

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
GENERATOR_DIR = PROJECT_ROOT / "generator"

# Ensure local modules are importable both on the driver and Ray workers
for path in (PROJECT_ROOT, GENERATOR_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

pythonpath_parts = [str(PROJECT_ROOT), str(GENERATOR_DIR)]
existing_pythonpath = os.environ.get("PYTHONPATH")
if existing_pythonpath:
    pythonpath_parts.append(existing_pythonpath)
os.environ["PYTHONPATH"] = ":".join(pythonpath_parts)

import ray

import torch
from generator.gen import get_operator_combination_for_level, get_sample

os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
ray.init(
    address="auto",
    runtime_env={
        # "working_dir": str(PROJECT_ROOT),
        "env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]},
    },
)

logger = logging.getLogger(__name__)


def configure_logger(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger.setLevel(logging.INFO)
    if any(
        isinstance(handler, logging.FileHandler)
        and handler.baseFilename == os.path.abspath(log_path)
        for handler in logger.handlers
    ):
        return
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@ray.remote
class QueueActor:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def get(self):
        if self.items:
            return self.items.pop(0)
        return None

    def qsize(self):
        return len(self.items)


@ray.remote(num_cpus=1)
class Worker:
    def __init__(
        self,
        input_queue,
        result_queue,
        level: int,
        max_oom_retries: int,
        max_timeout_retries: int,
        max_invalid_retries: int,
        sample_timeout_seconds: int,
        log_path: str,
        use_gpu: bool,
    ):
        configure_logger(log_path)
        self.logger = logging.getLogger(__name__)
        self.input_queue = input_queue
        self.result_queue = result_queue
        self.level = level
        self.max_oom_retries = max_oom_retries
        self.max_timeout_retries = max_timeout_retries
        self.max_invalid_retries = max_invalid_retries
        self.sample_timeout_seconds = sample_timeout_seconds
        self.use_gpu = use_gpu
        signal.signal(signal.SIGALRM, self._timeout_handler)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError("Sample generation timed out")

    def _generate_sample(self, job_idx: int) -> Dict:
        if not isinstance(job_idx, int):
            raise TypeError("job_idx must be an int")
        op_types = get_operator_combination_for_level(self.level, job_idx)
        if not op_types:
            raise ValueError(f"No operators generated for level {self.level}")
        num_nodes = len(op_types)
        gpu_id = torch.cuda.current_device() if self.use_gpu else None
        oom_attempts = 0
        timeout_attempts = 0
        while True:
            try:
                signal.alarm(self.sample_timeout_seconds)
                sample = get_sample(
                    2**34,
                    2**35,
                    num_nodes,
                    op_types,
                    self.level,
                )
                signal.alarm(0)
                return sample
            except torch.cuda.OutOfMemoryError:
                if not self.use_gpu:
                    raise
                oom_attempts += 1
                if self.use_gpu:
                    torch.cuda.empty_cache()
                if oom_attempts >= self.max_oom_retries:
                    raise
                self.logger.warning(
                    "GPU %s OOM at idx %s, retry %s/%s",
                    gpu_id,
                    job_idx,
                    oom_attempts,
                    self.max_oom_retries,
                )
            except TimeoutError as exc:
                timeout_attempts += 1
                signal.alarm(0)
                if self.use_gpu:
                    torch.cuda.empty_cache()
                if timeout_attempts >= self.max_timeout_retries:
                    raise exc
                self.logger.warning(
                    "Timeout at idx %s, retry %s/%s",
                    job_idx,
                    timeout_attempts,
                    self.max_timeout_retries,
                )
            except Exception:
                signal.alarm(0)
                if self.use_gpu:
                    torch.cuda.empty_cache()
                raise
        raise RuntimeError(f"Exceeded max retries for job_idx={job_idx}")

    def run(self):
        while True:
            item: Dict[str, int] = ray.get(self.input_queue.get.remote())
            if item is None:
                time.sleep(1.0)
                continue
            job_idx = item["job_idx"]
            try:
                sample = self._generate_sample(job_idx)
                self.result_queue.put.remote({"ok": True, "result": sample})
            except Exception as exc:
                self.logger.warning("Job %s failed: %s", job_idx, exc)
                self.result_queue.put.remote(
                    {"ok": False, "result": {"job_idx": job_idx, "error": str(exc)}}
                )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--target_count", type=int, required=True)
    parser.add_argument("--max_oom_retries", type=int, default=10)
    parser.add_argument("--max_timeout_retries", type=int, default=5)
    parser.add_argument("--max_invalid_retries", type=int, default=10)
    parser.add_argument("--sample_timeout_seconds", type=int, default=300)
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPUs for workers. If omitted, workers run on CPU only.",
    )
    args = parser.parse_args()

    output_file = args.output_file
    level = args.level
    target_count = args.target_count
    max_oom_retries = args.max_oom_retries
    max_timeout_retries = args.max_timeout_retries
    max_invalid_retries = args.max_invalid_retries
    sample_timeout_seconds = args.sample_timeout_seconds
    use_gpu = args.use_gpu

    if target_count <= 0:
        raise ValueError("target_count must be positive")
    done_fn = output_file + ".done"
    if os.path.isfile(done_fn):
        print(f"skip {done_fn}")
        return
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    log_path = os.path.join(os.path.dirname(output_file) or ".", "ray.log")
    configure_logger(log_path)

    existing_samples = []
    if os.path.isfile(output_file):
        with open(output_file, "r") as fid:
            for line in fid:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    start_idx = len(existing_samples)
    if start_idx >= target_count:
        logger.info("Found %s samples in %s, target already reached.", start_idx, output_file)
        return

    logger.info(
        "Resume generation: existing=%s target=%s remaining=%s",
        start_idx,
        target_count,
        target_count - start_idx,
    )

    input_queue = QueueActor.remote()
    result_queue = QueueActor.remote()

    resources = ray.cluster_resources()
    total_cpus = int(resources.get("CPU", 0))
    total_gpus = int(resources.get("GPU", 0))
    if use_gpu:
        if total_gpus <= 0:
            raise RuntimeError("No GPU resources available in Ray cluster")
        worker_count = total_gpus
    else:
        if total_cpus <= 0:
            raise RuntimeError("No CPU resources available in Ray cluster")
        worker_count = total_cpus
    logger.info(
        "Ray cluster resources: %s CPUs, %s GPUs (use_gpu=%s)",
        total_cpus,
        total_gpus,
        use_gpu,
    )

    worker_pool = [
        Worker.options(num_gpus=1 if use_gpu else 0).remote(
            input_queue,
            result_queue,
            level,
            max_oom_retries,
            max_timeout_retries,
            max_invalid_retries,
            sample_timeout_seconds,
            log_path,
            use_gpu,
        )
        for _ in range(worker_count)
    ]
    for worker in worker_pool:
        worker.run.remote()

    for idx in range(start_idx, target_count):
        input_queue.put.remote({"job_idx": idx})

    dumped_results = 0
    total_jobs = target_count - start_idx
    success_count = 0
    failed_count = 0
    start_time = time.time()
    next_progress_percent = 1
    with open(output_file, "a") as fout:
        while dumped_results < total_jobs:
            qsize: int = cast(int, ray.get(result_queue.qsize.remote()))
            if qsize > 0:
                item: Dict = ray.get(result_queue.get.remote())
                if item is None:
                    time.sleep(1.0)
                    continue
                dumped_results += 1
                if item.get("ok"):
                    fout.write(json.dumps(item["result"]) + "\n")
                    fout.flush()
                    success_count += 1
                else:
                    failed_count += 1
                progress_percent = int(dumped_results * 100 / total_jobs)
                while progress_percent >= next_progress_percent and next_progress_percent <= 100:
                    elapsed = time.time() - start_time
                    rate = dumped_results / elapsed if elapsed > 0 else 0.0
                    eta_seconds = (total_jobs - dumped_results) / rate if rate > 0 else 0
                    print(
                        f"{output_file}: generated {dumped_results}/{total_jobs} "
                        f"({next_progress_percent}%) ETA {_format_seconds(eta_seconds)}",
                        flush=True,
                    )
                    next_progress_percent += 1
            else:
                time.sleep(1.0)

    done_fn = output_file + ".done"
    with open(done_fn, "w") as fid:
        fid.write(
            json.dumps(
                {
                    "total_run_time": time.time() - start_time,
                    "success": success_count,
                    "failed": failed_count,
                    "target_count": target_count,
                    "existing_samples": start_idx,
                }
            )
        )
    logger.info(
        "Generation completed: wrote %s new samples (%s failed). Output: %s",
        success_count,
        failed_count,
        output_file,
    )


if __name__ == "__main__":
    main()