from typing import Any, Literal, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from typing_extensions import override

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LearningRateMonitorWrapper(LearningRateMonitor):
    """
    A wrapper around LearningRateMonitor that automatically disables itself
    when there are no loggers available, instead of raising an exception.

    This is useful for configurations where logging may be disabled but you
    still want to use the same callback configuration.

    Args:
        logging_interval: set to ``'epoch'`` or ``'step'`` to log ``lr`` of all optimizers
            at the same interval, set to ``None`` to log at individual interval
            according to the ``interval`` key of each scheduler. Defaults to ``None``.
        log_momentum: option to also log the momentum values of the optimizer, if the optimizer
            has the ``momentum`` or ``betas`` attribute. Defaults to ``False``.
        log_weight_decay: option to also log the weight decay values of the optimizer. Defaults to
            ``False``.

    Example::

        >>> from src.utils.lr_monitor_wrapper import LearningRateMonitorWrapper
        >>> from lightning.pytorch import Trainer
        >>> lr_monitor = LearningRateMonitorWrapper(logging_interval='step')
        >>> trainer = Trainer(callbacks=[lr_monitor])  # Works even without loggers
    """

    def __init__(
        self,
        logging_interval: Optional[Literal["step", "epoch"]] = None,
        log_momentum: bool = False,
        log_weight_decay: bool = False,
    ) -> None:
        super().__init__(
            logging_interval=logging_interval,
            log_momentum=log_momentum,
            log_weight_decay=log_weight_decay,
        )
        self._disabled = False

    @override
    def on_train_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        """
        Called before training. If no loggers are available, disable this callback
        instead of raising an exception.
        """
        if not trainer.loggers:
            log.info(
                "No loggers found. LearningRateMonitor will be disabled for this training run."
            )
            self._disabled = True
            return

        # Call parent implementation if loggers are available
        super().on_train_start(trainer, *args, **kwargs)

    @override
    def on_train_batch_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        """Skip if disabled."""
        if self._disabled:
            return
        super().on_train_batch_start(trainer, *args, **kwargs)

    @override
    def on_train_epoch_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        """Skip if disabled."""
        if self._disabled:
            return
        super().on_train_epoch_start(trainer, *args, **kwargs)
