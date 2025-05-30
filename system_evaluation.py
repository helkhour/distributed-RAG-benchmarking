import psutil
import os
import time
import torch
import logging

logger = logging.getLogger(__name__)

class SystemEvaluator:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.cpu_times_start = None
        # logger.debug(f"SystemEvaluator initialized for process PID: {os.getpid()}")

    def start_monitoring(self):
        """Start timing and CPU usage tracking."""
        self.start_time = time.time()
        self.cpu_times_start = self.process.cpu_times()
        # logger.debug(f"Monitoring started at {time.ctime(self.start_time)}")

    def log_resources(self, prefix):
        """Log CPU percentage, cumulative CPU time, memory usage, and GPU usage."""
        cpu_percent = self.process.cpu_percent(interval=1.0)
        cpu_times = self.process.cpu_times()
        cpu_time_total = cpu_times.user + cpu_times.system
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        logger.info(f"{prefix} - CPU: {cpu_percent:.2f}%, CPU Time: {cpu_time_total:.2f}s, "
                   f"Memory: {memory_mb:.2f} MB, GPU Memory: {gpu_memory:.2f} MB")
        # logger.debug(f"Detailed resources: cpu_times={cpu_times}, memory_info={self.process.memory_info()}")

    def end_monitoring(self, label):
        """End monitoring, log resources, and return duration and CPU delta."""
        if self.start_time is None or self.cpu_times_start is None:
            logger.error("Monitoring not started.")
            raise ValueError("Monitoring not started.")
        
        duration = time.time() - self.start_time
        cpu_times_end = self.process.cpu_times()
        cpu_time_delta = (cpu_times_end.user + cpu_times_end.system) - \
                        (self.cpu_times_start.user + self.cpu_times_start.system)
        
        cpu_percent = self.process.cpu_percent(interval=1.0)
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        logger.info(f"{label} - Duration: {duration:.2f}s, CPU Time Delta: {cpu_time_delta:.2f}s, "
                   f"CPU: {cpu_percent:.2f}%, Memory: {memory_mb:.2f} MB, GPU Memory: {gpu_memory:.2f} MB")
        # logger.debug(f"End monitoring details: cpu_times_end={cpu_times_end}, cpu_delta_breakdown={cpu_time_delta}")
        
        self.start_time = None
        self.cpu_times_start = None
        return duration, cpu_time_delta