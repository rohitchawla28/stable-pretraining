"""Check the benchmark pipeline on CPU without downloading CIFAR-10.

Run from the repository root::

    python benchmarks/cifar10/dcl-resnet18-test.py
    python benchmarks/cifar10/dcl-resnet18-test.py --method simclr

Uses the real ResNet-18, projector, augmentations, loss, optimizer, scheduler,
Manager, kNN and linear-probe callbacks, logger, and checkpoint writer. Four
backbone optimizer steps on synthetic images check execution, not representation
quality or convergence.
The method class and distributed correctness remain covered by the test suite.
"""

import runpy
import sys
from pathlib import Path

import lightning as pl
import torch


class SmokeChecks(pl.Callback):
    """Fail on invalid loss, gradients, embeddings, or a missing encoder update."""

    def on_train_start(self, trainer, pl_module):
        self.before = pl_module.backbone.conv1.weight.detach().cpu().clone()
        self.probe_before = (
            pl_module.callbacks_modules["linear_probe"].weight.detach().cpu().clone()
        )
        self.train_batches = 0
        self.val_batches = 0

    def on_after_backward(self, trainer, pl_module):
        for name, parameter in pl_module.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, f"Missing gradient: {name}"
                assert torch.isfinite(parameter.grad).all(), f"Invalid gradient: {name}"

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        assert torch.isfinite(outputs["loss"]), "Nonfinite training loss"
        self.train_batches += 1

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        assert outputs["embedding"].shape == (len(batch["label"]), 512)
        assert torch.isfinite(outputs["embedding"]).all()
        assert outputs["linear_probe_preds"].shape == (len(batch["label"]), 10)
        assert torch.isfinite(outputs["linear_probe_preds"]).all()
        assert "loss" not in outputs
        self.val_batches += 1

    def on_train_end(self, trainer, pl_module):
        assert self.train_batches == 4 and trainer.global_step == 4
        assert self.val_batches == 2
        assert not torch.equal(
            self.before, pl_module.backbone.conv1.weight.detach().cpu()
        )
        assert not torch.equal(
            self.probe_before,
            pl_module.callbacks_modules["linear_probe"].weight.detach().cpu(),
        )
        for key in ("eval/knn_probe_accuracy", "eval/linear_probe_top1"):
            accuracy = trainer.callback_metrics[key]
            assert torch.isfinite(accuracy) and 0 <= accuracy <= 1


if __name__ == "__main__":
    benchmark = runpy.run_path(str(Path(__file__).with_name("dcl-resnet18.py")))
    args = benchmark["parse_args"](
        [
            "--output-dir",
            "outputs/dcl-smoke",
            *sys.argv[1:],
            "--smoke",
        ]
    )
    trainer, module = benchmark["run_experiment"](args, callbacks=[SmokeChecks()])
    final_path = Path(trainer.logger.log_dir) / "checkpoints" / "final.ckpt"
    assert final_path.is_file(), "No final checkpoint was written"
    checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == args.epochs - 1
    assert checkpoint["global_step"] == trainer.global_step
    torch.testing.assert_close(
        checkpoint["state_dict"]["backbone.conv1.weight"],
        module.backbone.conv1.weight.detach().cpu(),
    )
    torch.testing.assert_close(
        checkpoint["state_dict"]["callbacks_modules.linear_probe.weight"],
        module.callbacks_modules["linear_probe"].weight.detach().cpu(),
    )
    assert (Path(trainer.logger.log_dir) / "metrics.csv").is_file(), (
        "Missing metrics CSV"
    )
    print(
        f"PASS: {args.method} completed training, evaluation, logging, and checkpointing."
    )
