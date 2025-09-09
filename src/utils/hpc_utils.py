import logging
import os
import signal
from pathlib import Path
from types import FrameType
from typing import Callable, List, Literal, Optional, Union

import lightning.pytorch as pl
import torch
import torch.distributed
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from omegaconf import DictConfig

# copied from signal.pyi
_SIGNUM = Union[int, signal.Signals]

log = logging.getLogger(__name__)

HPC_STATUS_TYPES = Literal["RUNNING", "STOPPED", "FINISHED", "ERROR"]


# Global status
_STATUS: HPC_STATUS_TYPES = "RUNNING"


@rank_zero_only
def _write_hpc_status(status: HPC_STATUS_TYPES, status_path: str) -> None:
    log.info(f"Writing status: {status} to file: {status_path}")
    with open(status_path, "w") as f:
        f.write(status)


def is_hpc() -> bool:
    """Check if the current environment is a HPC environment (e.g. SLURM)."""
    return "SLURM_JOBID" in os.environ


class DefaultTrainerSignalFunctor:
    """Default signal handler for the trainer."""

    def __init__(
        self,
        trainer: pl.Trainer,
        stop_signals: List[_SIGNUM] = [signal.SIGUSR1, signal.SIGUSR2],
        status_path: Optional[str] = None,
    ):
        if status_path is None:
            status_path = os.path.join(trainer.default_root_dir, "status.txt")
        self.trainer = trainer
        self.status_path = status_path
        self.stop_signals = stop_signals

    def __call__(self, signum: _SIGNUM, _: FrameType) -> None:
        # Saving checkpoint will be saved automatically by the default trainer signal handler.
        # Here, you only save the status for custom steps
        global _STATUS
        if signum in self.stop_signals:
            log.info(f"{self.__class__.__name__} received signal: {signum} on PID: {os.getpid()}")
            self.trainer.should_stop = True
            _STATUS = "STOPPED"
            _write_hpc_status("STOPPED", self.status_path)


class HPCSignalHandler:
    """Signal handler class to handle SIGUSR1 and SIGUSR2 signals.

    To run it with SLURM, use the following sbatch script that sends SIGUSR1 300
    seconds before the job ends:

    .. code-block:: bash

        #!/bin/bash
        #SBATCH --time=10:00
        #SBATCH --signal=USR1@300
        ...
    """

    _ENTERED: bool = False

    def __init__(
        self,
        trainer: pl.Trainer,
        signal_handler: Optional[Callable] = None,
        status_path: Optional[Union[str, Path]] = None,
        stop_signals: List[_SIGNUM] = [signal.SIGUSR1, signal.SIGUSR2],
    ):
        self.trainer = trainer
        if signal_handler is None:
            signal_handler = DefaultTrainerSignalFunctor(trainer, stop_signals=stop_signals)
        self._signal_handler = signal_handler

        # Get distributed backend and move the _SIG_RECEIVED to the device
        # Global flag for signaling for all DDP processes
        self._DISTRIBUTED = torch.distributed.is_initialized()
        self._RANK = torch.distributed.get_rank() if self._DISTRIBUTED else 0

        # Register signal handlers
        self._register_sigusr_handler(stop_signals=stop_signals)

        # Save the status of the current run
        if status_path is None:
            status_path = os.path.join(trainer.default_root_dir, "status.txt")

        self.status_path = status_path
        if self._RANK == 0:
            os.makedirs(os.path.dirname(status_path), exist_ok=True)

        self.STATUS = "RUNNING"

    @property
    def STATUS(self):
        """Get the status of the current run."""
        global _STATUS
        return _STATUS

    @STATUS.setter
    def STATUS(self, status: str):
        """Set the status of the current run."""
        _write_hpc_status(status, self.status_path)
        global _STATUS
        _STATUS = status

    def __enter__(self):
        self._ENTERED = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            log.error(f"Exception: {exc_type} - {exc_value}")
            self.STATUS = "ERROR"

        # When it goes out of context without stopping, or error, then it is finished.
        if not self.is_stopped():
            self.STATUS = "FINISHED"

    def _register_sigusr_handler(self, stop_signals: List[_SIGNUM]):
        # Register signal handlers
        for sig in stop_signals:
            signal.signal(sig, self._signal_handler)

    def is_stopped(self):
        global _STATUS
        return _STATUS == "STOPPED"

    def is_finished(self):
        global _STATUS
        return _STATUS == "FINISHED"


def hpc_config(cfg: DictConfig) -> None:
    """Modify the configuration for HPC environment before running an experiment."""
    if not is_hpc():
        return cfg

    # Set the output_dir to be SLURM job ID
    job_id = os.environ["SLURM_JOBID"]
    # If os.environ["CKPT_SLURM_JOBID"] is set, use it instead
    if "CKPT_SLURM_JOBID" in os.environ:
        job_id = os.environ["CKPT_SLURM_JOBID"]
    cfg.paths.output_dir = os.path.join("results", job_id)

    # Make sure the output_dir exists
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    # Check if the checkpoints exist
    checkpoint_dir = os.path.join(cfg.paths.output_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        latest_checkpoint = max(
            checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x))
        )
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        if os.path.exists(checkpoint_path):
            cfg.ckpt_path = checkpoint_path

    # Replace the logger.wandb._target_ to AutoResumeWandbLogger
    if "debug" not in cfg.task_name and "wandb" in cfg.logger:
        # cfg.logger.wandb._target_ = "src.utils.autoresume_wandb_logger.AutoResumeWandbLogger"
        # Print warning if not using AutoResumeWandbLogger
        if (
            str(cfg.logger.wandb._target_)
            != "src.utils.auto_resume_wandb_logger.AutoResumeWandbLogger"
        ):
            log.warning(
                "Using WandbLogger without AutoResumeWandbLogger. "
                "Please make sure to resume the run manually."
            )
