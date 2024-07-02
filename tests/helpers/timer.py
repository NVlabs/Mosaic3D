import multiprocessing as mp
import subprocess
import time
from typing import Optional

import numpy as np
import torch


def get_gpu_memory_usage(device_id: int = 0):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().split("\n")[device_id]


class Meter:
    def __init__(
        self,
        name: Optional[str] = None,
        interval_second: float = 0.1,
        device_id: int = 0,
    ):
        self.name = name
        self.queue = mp.Queue()
        self.stop_event = mp.Event()
        self.interval = interval_second
        self.min_time = float("inf")
        self.max_memory_usage = 0
        self.start_memory_usage = 0
        self.device_id = device_id

    def _collect_gpu_memory_usage(self):
        """Collects NVIDIA GPU memory usage and sends it to the main process through a queue."""
        while not self.stop_event.is_set():
            memory_usage = int(
                get_gpu_memory_usage(device_id=self.device_id)
            )  # Directly get the memory usage
            self.queue.put(memory_usage)
            time.sleep(self.interval)

    def __enter__(self):
        # Reset the stop event in case this is a reuse of the instance
        self.stop_event.clear()
        # Recreate the process to ensure it's fresh and runnable
        self.process = mp.Process(target=self._collect_gpu_memory_usage)
        self.process.start()
        self.start_time = time.time()
        # Reset max_memory_usage for a fresh start
        self.start_memory_usage = int(get_gpu_memory_usage())
        self.max_memory_usage = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate the time taken
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.elapsed_time < self.min_time:
            self.min_time = self.elapsed_time

        # Signal the worker process to stop and wait for it to finish
        self.stop_event.set()
        self.process.join()

        # Collect and find the maximum memory usage
        while not self.queue.empty():
            memory_usage = self.queue.get()
            if memory_usage > self.max_memory_usage:
                self.max_memory_usage = memory_usage

        # Optionally reset or prepare the instance for potential reuse here
        return False  # Don't suppress exceptions

    def __del__(self):
        # Ensure the process is not running
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.process.join()

        # Close and join the queue
        self.queue.close()
        self.queue.join_thread()
