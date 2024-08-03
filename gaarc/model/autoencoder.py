import pytorch_lightning as pl
import segmentation_models_pytorch as smp  # type: ignore[import-untyped]
import torch
import torch.nn as nn

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
    encoder_first_block_channels : int
        Size of the channel dimension in the first processing block.
        The rest of the channels are inferred from this initial value.
    model_depth : int
        Amount of blocks the encoder and the decoder will be composed of.
    verbose_training : bool
        If True, will print information through the terminal during training.
        By default, False.
    """

    def __init__(
        self,
        input_channels: int,
        number_of_classes: int,
        encoder_first_block_channels: int,
        model_layer_depth: int,
        verbose_training: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._input_channels: int = input_channels
        self._model_layer_depth: int = model_layer_depth
        self._verbose_training: bool = verbose_training
        self._epochs_trained: int = 0
        self._step_outputs: dict[list] = {"train": [], "valid": [], "test": []}

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
        sample = batch[0]
        target = batch[1].contiguous().long()

        prediction = self.forward(sample)

        # Padding removal
        padding = batch[2]
        prediction_without_padding = prediction[
            :, :, padding[0] : -padding[1], padding[2] : -padding[3]
        ]

        loss = self.loss_fn(prediction_without_padding, target)

        prediction_classes = torch.max(prediction_without_padding, dim=1)[1]

        tp, fp, fn, tn = smp.metrics.get_stats(
            prediction_classes, target, mode="multiclass", num_classes=10
        )

        output_metrics = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        self._step_outputs[stage].append(output_metrics)

        return loss

    def on_epoch_end(self, stage):
        total_loss = 0
        iter_count = len(self._step_outputs[stage])

        for idx in range(iter_count):
            total_loss += self._step_outputs[stage][idx]["loss"].item()

        if self._step_outputs[stage]:
            tp = torch.cat([output["tp"] for output in self._step_outputs[stage]])
            fp = torch.cat([output["fp"] for output in self._step_outputs[stage]])
            fn = torch.cat([output["fn"] for output in self._step_outputs[stage]])
            tn = torch.cat([output["tn"] for output in self._step_outputs[stage]])

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

            if self._verbose_training:
                if stage == "train" or stage == "valid":
                    if stage == "train":
                        print_color = "\033[94m"
                    elif stage == "valid":
                        print_color = "\033[92m"
                print(
                    f"{print_color}Epoch {self._epochs_trained:02d} {stage}: {metrics}"
                )

            if stage == "train":
                self._epochs_trained += 1

            self._step_outputs[stage].clear()

            self.log_dict(metrics, prog_bar=True)

    def training_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ, unused-argument
        return self.step(batch, "train")

    def on_train_epoch_end(self):
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
