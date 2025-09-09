import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning_utilities.core.imports import RequirementCache

if TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run


_WANDB_AVAILABLE = RequirementCache("wandb>=0.12.10")
log = logging.getLogger(__name__)


class AutoResumeWandbLogger(WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        Logger.__init__(self)

        if not _WANDB_AVAILABLE:
            raise ModuleNotFoundError(str(_WANDB_AVAILABLE))

        if offline and log_model:
            raise MisconfigurationException(
                f"Providing log_model={log_model} and offline={offline} is an invalid configuration"
                " since model checkpoints cannot be uploaded in offline mode.\n"
                "Hint: Set `offline=False` to log your model."
            )

        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._experiment = experiment
        self._logged_model_time: Dict[str, float] = {}
        self._checkpoint_callback: Optional[ModelCheckpoint] = None

        # paths are processed as strings
        if save_dir is not None:
            save_dir = os.fspath(save_dir)

        project = project or os.environ.get("WANDB_PROJECT", "lightning_logs")

        import wandb

        if id is None:
            id = wandb.util.generate_id()

        # if wandb_id file exists, set id and resume to "must"
        wandb_id_file = os.path.join(save_dir, "wandb_id.txt")
        resume, id = self._resume_wandb_id(wandb_id_file, id)

        # set wandb init arguments
        self._wandb_init: Dict[str, Any] = {
            "name": name,
            "project": project,
            "dir": save_dir or dir,
            "id": version or id,
            "resume": resume,
            "anonymous": ("allow" if anonymous else None),
        }
        self._wandb_init.update(**kwargs)
        # extract parameters
        self._project = self._wandb_init.get("project")
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
        self._checkpoint_name = checkpoint_name
        # Add the $SLURM_JOBID to the wandb run name and tag
        if "SLURM_JOBID" in os.environ:
            job_id = os.environ["SLURM_JOBID"]
            if "CKPT_SLURM_JOBID" in os.environ:
                job_id = os.environ["CKPT_SLURM_JOBID"]

            # If name is not None, append the SLURM_JOBID to the name. Otherwise, set the name to SLURM_JOBID
            if self._name is not None:
                self._wandb_init["name"] = f"{self._name}_{job_id}"
            else:
                self._wandb_init["name"] = job_id
            # append to the end of the tags list
            self._wandb_init["tags"] = [
                *self._wandb_init.get("tags", []),
                job_id,
            ]

    def _resume_wandb_id(self, wandb_id_file: str, id: str) -> tuple[str, str]:
        # Only the rank 0 will be used for initializing the experiment property and wandb.init. Other rank's variables will be discarded
        resume = "allow"
        if rank_zero_only.rank > 0:
            return resume, id

        if os.path.exists(wandb_id_file):
            with open(wandb_id_file) as f:
                id = f.read().strip()
            resume = "must"
            log.info(f"wandb_id found, setting wandb_id: {id}")
        else:
            with open(wandb_id_file, "w") as f:
                f.write(id)
            log.info(f"wandb_id not found in config, generating wandb_id: {id}")
        return resume, id
