import copy

import pytorch_lightning as pl
import torch
import xxhash  # pylint: disable=import-error
from torch import nn
from torch.nn.modules.loss import MSELoss

from gaarc.model.secondary_task_modules import STM, ARCSample, Loss
from gaarc.model.unet import UNet


class UNetTaskSolver(pl.LightningModule):
    """
    Pytorch lightning wrapper for the task solver which is intended to make use of the unet backbone
    from a pretrained UnetAutoencoder as backbone.
    """

    def __init__(
        self,
        autoencoder_backbone: nn.Module,
        initial_learning_rate: float = 0.0001,
        secondary_task_modules: list[STM] | None = None,
        secondary_tasks_global_loss_weight: float = 1.0,
        cache_secondary_task_samples: bool = True,
        verbose_training: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._initial_learning_rate: float = initial_learning_rate
        self._secondary_tasks_global_loss_weight = secondary_tasks_global_loss_weight
        self._sample_cache: dict[int, ARCSample] = {}
        self._cache_stm_samples: bool = cache_secondary_task_samples
        self._verbose_training: bool = verbose_training
        self._secondary_task_modules: nn.ModuleList = (
            nn.ModuleList(secondary_task_modules) if secondary_task_modules else nn.ModuleList([])
        )
        self._epochs_trained: int = -1
        self._step_outputs: dict[list] = {"train": [], "valid": [], "test": []}
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self._model: nn.Module = autoencoder_backbone

        self._prediction_encoder = self._model.encoder
        self._target_encoder = copy.deepcopy(self._model.encoder)

        self.loss_fn = MSELoss()

        self.save_hyperparameters()

    @property
    def model(self) -> UNet:
        return self._model

    def forward(self, image):  # pylint: disable=arguments-differ
        mask = self.model(image)

        return mask

    def step(self, batch, stage):
        inputs = batch[0]
        inputs_padding = batch[1]
        outputs = batch[2]
        outputs_padding = batch[3]

        predicted_samples = self.forward(inputs)
        predicted_samples = torch.max(predicted_samples, dim=1)[1].unsqueeze(1).type(torch.float)
        prediction_latent_vector = self._prediction_encoder(predicted_samples)[0]

        target_latent_vector = self._target_encoder(outputs)[0]

        loss = self.loss_fn(prediction_latent_vector, target_latent_vector)

        output_metrics = {
            "loss": loss,
        }

        self._step_outputs[stage].append(output_metrics)

        return loss

    def on_epoch_end(self, stage):
        total_loss = 0
        iter_count = len(self._step_outputs[stage])

        for idx in range(iter_count):
            total_loss += self._step_outputs[stage][idx]["loss"].detach()

        metrics = {
            f"{stage}_loss": total_loss / iter_count,
        }

        if stage == "valid":
            self._epochs_trained += 1

        if self._verbose_training:
            if stage == "train":
                print_color = "\033[94m"
            elif stage == "valid":
                print_color = "\033[92m"
            elif stage == "test":
                print_color = "\033[96m"

            print(f"{print_color}Epoch {self._epochs_trained:02d} {stage}: {metrics}")

        self._step_outputs[stage].clear()

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ, unused-argument
        return self.step(batch, "train")

    def on_train_epoch_end(self):
        return self.on_epoch_end("train")

    def validation_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ, unused-argument
        with torch.no_grad():
            return self.step(batch, "valid")

    def on_validation_epoch_end(self):
        with torch.no_grad():
            return self.on_epoch_end("valid")

    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ, unused-argument
        with torch.no_grad():
            return self.step(batch, "test")

    def on_test_epoch_end(self):
        with torch.no_grad():
            return self.on_epoch_end("test")

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

    def _get_arc_sample(self, sample_tensor: torch.Tensor) -> ARCSample:
        sample_array = sample_tensor.squeeze(0).numpy()
        hash_ = xxhash.xxh64()

        if self._cache_stm_samples:
            hash_.update(sample_array)
            sample_hash = hash_.intdigest()

            if sample_hash in self._sample_cache:
                sample = self._sample_cache[sample_hash]
            else:
                sample = ARCSample(sample_array)
                self._sample_cache[sample_hash] = sample

        else:
            sample = ARCSample(sample_array)

        return sample

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._initial_learning_rate)
