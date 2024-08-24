import pytorch_lightning as pl
import segmentation_models_pytorch as smp  # type: ignore[import-untyped]
import torch
import torch.nn as nn

from gaarc.model.secondary_task_modules import STM, Loss
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
    initial_learning_rate : float
        Initial value the optimizer's learning rate will be set to.
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
        initial_learning_rate: float = 0.0001,
        secondary_tasks_loss_scale: float = 0.1,
        verbose_training: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._input_channels: int = input_channels
        self._model_layer_depth: int = model_layer_depth
        self._initial_learning_rate: float = initial_learning_rate
        self._secondary_tasks_loss_scale = secondary_tasks_loss_scale
        self._verbose_training: bool = verbose_training
        self._secondary_task_modules: nn.ModuleList = nn.ModuleList([])
        self._epochs_trained: int = -1
        self._step_outputs: dict[list] = {"train": [], "valid": [], "test": []}
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

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
        samples = batch[0].to(self._device)
        paddings = batch[1]

        predictions = self.forward(samples)

        for sample, padding, prediction in zip(samples, paddings, predictions):
            target = self._crop_tensor(sample, padding).long()
            loss_target = target.unsqueeze(0).contiguous()

            prediction_without_padding = self._crop_tensor(
                prediction, padding
            ).unsqueeze(0)

            loss = self.loss_fn(prediction_without_padding, loss_target)

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

        if stage == "train" and self._secondary_task_modules:
            secondary_task_losses: list[Loss] = []

            for secondary_task_module in self._secondary_task_modules:
                secondary_task_loss = secondary_task_module.train_on_task(samples)

                for step_output in self._step_outputs[stage]:
                    step_output.update(
                        {f"{secondary_task_module.name} loss": secondary_task_loss}
                    )

                secondary_task_losses.append(secondary_task_loss)

            secondary_tasks_combined_loss = sum(secondary_task_losses)
            secondary_tasks_combined_loss *= self._secondary_tasks_loss_scale
            secondary_tasks_combined_loss /= len(self._secondary_task_modules)

            loss += secondary_tasks_combined_loss

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
                f"{stage}_precision": precision.item(),
                f"{stage}_recall": recall.item(),
                f"{stage}_accuracy": accuracy.item(),
                f"{stage}_f1_score": f1_score.item(),
                f"{stage}_per_image_iou": per_image_iou.item(),
                f"{stage}_dataset_iou": dataset_iou.item(),
            }

            if stage == "valid":
                self._epochs_trained += 1

            if stage == "train":
                if self._secondary_task_modules:
                    for secondary_task_module in self._secondary_task_modules:
                        secondary_task_loss_name = f"{secondary_task_module.name} loss"
                        secondary_task_loss = 0

                        for step_output in self._step_outputs[stage]:
                            secondary_task_loss += step_output[
                                secondary_task_loss_name
                            ].item()

                        metrics.update(
                            {secondary_task_loss_name: secondary_task_loss / iter_count}
                        )

            if self._verbose_training:
                if stage == "train" or stage == "valid":
                    if stage == "train":
                        print_color = "\033[94m"
                    elif stage == "valid":
                        print_color = "\033[92m"
                print(
                    f"{print_color}Epoch {self._epochs_trained:02d} {stage}: {metrics}"
                )

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

    def add_secondary_task_module(
        self, secondary_task_module: STM, sample_example: torch.Tensor, **kwargs
    ) -> None:
        """
        Adds an STM class to the secondary tasks to be used during trained.

        This method expects the class, not an initialized instance.
        It will be initialized based on the architecture of the autoencoder and the
        size of the provided sample.

        Parameters
        ----------
        secondary_task_module : STM
            Uninitialized STM
        """
        if len(sample_example.shape) == 3:
            sample_example = sample_example.unsqueeze(0)

        latent_space_size = self.model.encoder(sample_example)[0].shape[1]

        secondary_task_module_instance = secondary_task_module(
            self.model.encoder, latent_space_size, **kwargs
        )

        self._secondary_task_modules.append(secondary_task_module_instance)

    def _crop_tensor(
        self, tensor: torch.Tensor, cropping_per_side: tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
        Crops a 4D tensor (representing a batch of images) [B, C, H, W], selecting
        how much to remove from each side.

        The values provided in the tuple cropping_per_side represents, in respective
        order, top, bottom, left, right.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to crop.
        cropping_per_side : tuple[int, int, int, int]
            How much positions crop from each side, in respective order, in respective
            from order, top, bottom, left, right.


        Returns
        -------
        torch.Tensor
            Cropped tensor.
        """
        cropped_tensor = tensor[
            :,
            cropping_per_side[0] : -cropping_per_side[1],
            cropping_per_side[2] : -cropping_per_side[3],
        ]

        return cropped_tensor

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._initial_learning_rate)
