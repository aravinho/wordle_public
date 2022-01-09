import os
from typing import Any, Dict, List, Optional

from torch.utils.tensorboard import SummaryWriter

from wordle.policies.policy_gradient import io_utils


class ExperimentLogger:
    def __init__(
        self,
        log_dir: str,
        print_every: int,
        log_every: int = 1,
        log_tb: bool = True,
        tb_writer_names: Optional[List[str]] = None,
        default_tb_writer_name: Optional[str] = None,
    ):
        self.log_dir = io_utils.ensure_nonexistent_and_create_dir(log_dir)

        # Tensorboard
        self.log_tb = log_tb
        if self.log_tb:
            if tb_writer_names is None:
                self.tb_summary_writer = SummaryWriter(self.log_dir)
            else:
                self.tb_summary_writer = None
                # Multiple writers (e.g. train and val)
                self.tb_summary_writers = {}
                for writer_name in tb_writer_names:
                    writer_dir = io_utils.ensure_nonexistent_and_create_dir(
                        os.path.join(self.log_dir, writer_name)
                    )
                    self.tb_summary_writers[writer_name] = SummaryWriter(writer_dir)

                # Optionally set a default tb writer
                if default_tb_writer_name is not None:
                    self.tb_summary_writer = self.tb_summary_writers[
                        default_tb_writer_name
                    ]

        self.print_every = print_every
        self.log_every = log_every

    def _get_tb_writer(self, tb_writer_name):
        if tb_writer_name is None:
            assert self.tb_summary_writer is not None
            writer = self.tb_summary_writer
        else:
            writer = self.tb_summary_writers[tb_writer_name]

        return writer

    def log_scalars(
        self,
        scalars: Dict[str, Any],
        step: int,
        tb_writer_name: Optional[str] = None,
        prefix: str = "",
        force_log: bool = False,
        keys_to_print: List[str] = [],
    ):

        do_log = force_log or (step % self.log_every == 0)
        if not do_log:
            return

        # Optinally save to tb
        if self.log_tb:

            # Determine which writer to use
            writer = self._get_tb_writer(tb_writer_name)

            # write each scalar
            for tag, scalar_value in scalars.items():
                writer.add_scalar(
                    tag=f"{prefix}{tag}", scalar_value=scalar_value, global_step=step
                )
                writer.flush()

        # Optionally print
        if step % self.print_every == 0 and len(keys_to_print) > 0:
            print(
                "================================================================\n"
                + f"Metrics at step {step}:"
            )
            for k, v in scalars.items():
                if k in keys_to_print:
                    print(f"{k}: {v}")
            print("================================================================\n")
