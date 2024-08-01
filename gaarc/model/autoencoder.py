import pytorch_lightning as pl
import segmentation_models_pytorch as smp  # type: ignore[import-untyped]
import torch
import torch.nn as nn
import torch.nn.functional as F

from gaarc.model.unet import UNet


class UNetAutoEncoder(pl.LightningModule):
    """
    Pytorch lightning wrapper for the AutoEncoder application of the U-Net model.
    It provides an interface for training and inference.

    Shares it's input arguments with the UNet model, and it will initialise
    the model.

    Parameters
    ----------
     input_channels : int
        Size of the channel dimension of the input data.
        Can be 3 for rgb images or 1 for data shaped like a matrix.
    output_channels : int
        Size of the channel dimension of the output data.
        This is analog to number of classes predicted by the last convolutional layer.
    encoder_first_block_channels: int
        Size of the channel dimension in the first processing block.
        The rest of the channels are inferred from this initial value.
    model_depth: int
        Amount of blocks the encoder and the decoder will be composed of.
    """

    def __init__(
        self,
        input_channels: int,
        number_of_classes: int,
        encoder_first_block_channels: int,
        model_layer_depth: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._input_channels: int = input_channels
        self._model_layer_depth: int = model_layer_depth
        self.epochs_trained: int = 0
        self.step_outputs: list = []

        self._model: nn.Module = UNet(
            input_channels,
            number_of_classes,
            encoder_first_block_channels,
            model_layer_depth,
        )

        self.loss_fn = smp.losses.DiceLoss(
            "multiclass",
            from_logits=True,
        )

        self.save_hyperparameters()

    @property
    def model(self) -> UNet:
        return self._model

    def forward(self, image):  # pylint: disable=arguments-differ
        mask = self.model(image)

        return mask

    def step(self, batch, stage):
        # X retrival
        sample = batch[0]

        # Y retrival
        target = batch[1].contiguous()

        # Prediction
        prediction = self.forward(sample)

        # Padding removal
        padding = batch[2]
        prediction_without_padding = prediction[
            :, :, padding[0] : -padding[1], padding[2] : -padding[3]
        ]

        # Loss calculation
        loss = self.loss_fn(prediction_without_padding, target.long())

        # Target preparation for other metrics
        target = (
            # pylint: disable=not-callable
            F.one_hot(target.long(), 10)
            .permute(0, 3, 1, 2)
            .long()
        )

        # Confussion matrix components calculation
        tp, fp, fn, tn = smp.metrics.get_stats(
            prediction_without_padding.long(), target, mode="multiclass", num_classes=10
        )

        output_metrics = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        self.step_outputs.append(output_metrics)

        return loss

    def on_epoch_end(self, stage):
        total_loss = 0
        iter_count = len(self.step_outputs)

        for idx in range(iter_count):
            total_loss += self.step_outputs[idx]["loss"].item()

        if self.step_outputs:
            tp = torch.cat([output["tp"] for output in self.step_outputs])
            fp = torch.cat([output["fp"] for output in self.step_outputs])
            fn = torch.cat([output["fn"] for output in self.step_outputs])
            tn = torch.cat([output["tn"] for output in self.step_outputs])

            per_image_iou = smp.metrics.iou_score(
                tp, fp, fn, tn, reduction="micro-imagewise"
            )

            dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
            precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")

            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")

            metrics = {
                f"{stage}_loss": total_loss / iter_count,
                f"{stage}_precision": precision,
                f"{stage}_recall": recall,
                f"{stage}_accuracy": accuracy,
                f"{stage}_f1_score": f1_score,
                f"{stage}_per_image_iou": per_image_iou,
                f"{stage}_dataset_iou": dataset_iou,
            }

            print(f"Epoch {self.epochs_trained:02d} completed: {metrics}")
            self.epochs_trained += 1

            self.step_outputs.clear()

            self.log_dict(metrics, prog_bar=True)

    def training_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ, unused-argument
        return self.step(batch, "train")

    def on_training_epoch_end(self):
        return self.on_epoch_end("train")

    def validation_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ, unused-argument
        return self.step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.on_epoch_end("valid")

    def test_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ, unused-argument
        return self.step(batch, "test")

    def on_test_epoch_end(self):
        return self.on_epoch_end("test")

    def configure_optimizers(self, learning_rate=0.0001):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
