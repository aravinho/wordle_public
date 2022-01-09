"""General deep learning pytorch utils."""
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import nn


def resolve_device(device: Optional[Union[torch.device, str]] = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        raise TypeError(f"Cannot resolve device {device} of type {type(device)}")

    return device


def make_optimizer(name, parameters, **kwargs):
    """Construct an optimizer based on a string identifier."""
    if name.lower() == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    else:
        raise NotImplementedError(f"optimizer {name} not implemented.")


def make_lr_scheduler(name, optimizer, **kwargs):
    if name.lower() == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, **kwargs)
    else:
        raise NotImplementedError(f"lr_scheduler {name} not implemented")


def make_activation(activation):
    if activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "tanh":
        return nn.Tanh()
    elif activation.lower() == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError(f"Unsupported activation {activation}")


class ScaleLayer(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, input):
        return self.val * input


class BatchFirstLSTMWithOutputProj(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        output_dim: Optional[int] = None,
        **lstm_kwargs,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            **lstm_kwargs,
        )

        self.output_dim = output_dim
        if output_dim is not None:
            self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, inputs, hidden_cell=None):
        lstm_outs, (final_hidden, final_cell) = self.lstm(inputs, hidden_cell)
        B, N, _ = lstm_outs.shape
        if self.output_dim is not None:
            lstm_outs = self.output_proj(lstm_outs.reshape(B * N, -1)).reshape(
                B, N, self.output_dim
            )

        return lstm_outs, (final_hidden, final_cell)


def make_mlp(
    input_dim,
    output_dim,
    hidden_sizes,
    activation="relu",
    final_activation=None,
    penultimate_layer_scale=None,
):

    mlp_layers = []
    layer_input_dim = input_dim
    num_layers = len(hidden_sizes) + 1
    for layer_idx in range(num_layers):
        if layer_idx == num_layers - 1:
            layer_output_dim = output_dim
        else:
            layer_output_dim = hidden_sizes[layer_idx]

        mlp_layers.append(nn.Linear(layer_input_dim, layer_output_dim))

        if layer_idx < num_layers - 1:
            mlp_layers.append(make_activation(activation))

            # If it's the penultimate layer, apply a scale
            if layer_idx == num_layers - 2 and penultimate_layer_scale is not None:
                mlp_layers.append(ScaleLayer(penultimate_layer_scale))

        elif layer_idx == num_layers - 1 and final_activation is not None:
            mlp_layers.append(make_activation(final_activation))

        layer_input_dim = layer_output_dim  # for next iteration

    return nn.Sequential(*mlp_layers)


def identity(x):
    return x


def torchify(arr, device, dtype=torch.float32):
    if isinstance(arr, list):
        arr = np.asarray(arr)
    return torch.from_numpy(arr).type(dtype).to(device)


def compute_accuracy(logits, labels, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = labels.size(0)

    topk_vals, topk_idxs = logits.topk(k=maxk, dim=1, largest=True, sorted=True)
    assert topk_vals.shape == topk_idxs.shape == (batch_size, maxk)

    correct = topk_idxs.eq(labels.unsqueeze(-1))

    res = []
    for k in topk:
        num_correct_k = correct[:, :k].sum(dim=1).float().sum(dim=0)
        perc_correct_k = num_correct_k * (100.0 / batch_size)
        res.append(perc_correct_k)

    return res


class AlexNet(nn.Module):
    """Copied from pytorch, but flexible to variable numbers of input dim."""

    def __init__(self, input_channels, output_dim) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def clip_and_rescale(
    value, clip_min=-np.inf, clip_max=np.inf, rescale_min=-1, rescale_max=1
):
    """Convert from one range to another.

    Parameters
    ----------
    value : np.array
    clip_min : float | np.array
        Values below this are clipped up to this limit.
    clip_max : float | np.array
        Values above this are clipped down to this limit.
    squeeze_min : float | np.array
        The clipped values are then rescaled into a range with this lower bound.
    squeeze_max: float | np.array
        The clipped values are then rescaled into a range with this upper bound.

    Returns
    -------
    clipped_and_rescaled : np.array

    """

    # clip first
    try:
        clipped = value.clip(clip_min, clip_max)
    except TypeError:
        clipped = value
        clipped = torch.where(clipped > clip_min, clipped, clip_min)
        clipped = torch.where(clipped < clip_max, clipped, clip_max)
    # clipped = value.clip(clip_min, clip_max)
    # then find where on the spectrum
    point_on_01_spectrum = (clipped - clip_min) / (clip_max - clip_min)
    # then squeeze into range
    clipped_and_rescaled = rescale_min + (
        point_on_01_spectrum * (rescale_max - rescale_min)
    )
    return clipped_and_rescaled


def apply_weights(
    losses: Dict[str, torch.Tensor],
    weights: Dict[str, torch.Tensor],
    ref_tag: Optional[str] = None,
):
    tags = sorted(list(losses.keys()))

    if ref_tag is None:
        ref_tag = tags[0]
    ref_loss = losses[ref_tag]
    ref_weight = weights[ref_tag]
    assert weights[ref_tag] == 1

    eps = 1e-8
    proper_weights = {}
    for i, tag in enumerate(tags):

        tag_loss = losses[tag]
        tag_weight = weights[tag]
        # Determine proper scale ratio
        proper_scale = (tag_weight * ref_loss) / (ref_weight * tag_loss + eps)
        proper_weights[tag] = proper_scale

    final_loss = torch.zeros(1).mean().to(ref_loss.device)
    for tag in tags:
        final_loss += proper_weights[tag] * losses[tag]

    final_loss = final_loss.mean()

    return final_loss, proper_weights
