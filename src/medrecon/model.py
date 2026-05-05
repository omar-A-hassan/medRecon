import segmentation_models_pytorch as smp
from .config import ExperimentConfig


def build_model(config: ExperimentConfig) -> smp.Unet:
    return smp.Unet(
        encoder_name=config.encoder,
        encoder_weights=config.encoder_weights,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
