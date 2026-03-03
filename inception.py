# Reference
# https://github.com/pytorch/vision/blob/afc54f754c734d903a06194e416495e20d920ff6/torchvision/models/inception.py
# https://github.com/mseitzer/pytorch-fid/blob/b9c18118d082cbd263c1b8963fc4221dc1cbb659/src/pytorch_fid/inception.py
#
# original Tensorflow Inception model definition from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# `mseitzer/pytorch-fid` translate this weight to PyTorch version, which relies on `torchvision`'s InceptionV3 implementation, with some modifications to match the original Tensorflow Inception model definition
# this version supports spatial Fréchet Inception Distance & Inception Score evaluation, and remove model dependency on `torchvision`

import os.path as osp
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Any, Optional, Tuple

# Inception weights ported to Pytorch from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"


class InceptionV3(nn.Module):
    def __init__(self, inception_weights: Optional[str] = None) -> None:
        super().__init__()
        
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.Mixed_5b = FIDInceptionA(192, pool_features=32)
        self.Mixed_5c = FIDInceptionA(256, pool_features=64)
        self.Mixed_5d = FIDInceptionA(288, pool_features=64)
        
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
        self.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
        self.Mixed_6d = FIDInceptionC(768, channels_7x7=160)  # sfid spatial feature
        self.Mixed_6e = FIDInceptionC(768, channels_7x7=192)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = FIDInceptionE(1280, pool_type="avg")
        self.Mixed_7c = FIDInceptionE(2048, pool_type="max")

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 1008)

        load_kwargs = {"map_location": "cpu", "weights_only": True}
        if inception_weights is None or not osp.isfile(inception_weights):
            model_dir = None if inception_weights is None else osp.dirname(inception_weights)
            file_name = None if inception_weights is None else osp.basename(inception_weights)
            state_dict = torch.hub.load_state_dict_from_url(
                url=FID_WEIGHTS_URL,
                model_dir=model_dir,
                progress=True,
                check_hash=True,
                file_name=file_name,
                **load_kwargs
            )
        else:
            state_dict = torch.load(inception_weights, **load_kwargs)  # load from local
        self.load_state_dict(state_dict, strict=True)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # resize
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # normalize (0, 1) -> (-1, 1)
        x = 2 * x - 1

        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)

        # sfid feature map, spatial -> N x 192 x 17 x 17
        spatial = x.clone()
        spatial = self.Mixed_6d.branch1x1(spatial)

        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)

        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)

        # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L91-L93
        score = x.clone()  # (N, 2048, 1, 1)
        score = torch.flatten(score, 1)  # (N, 2048)
        # don't add bias, only multiply weight matrix
        score = score @ self.fc.weight.t()  # (N, 1008)
        score = F.softmax(score, dim=1)
        
        # (N, 2048, 1, 1), (N, 192, 17, 17), (N, 1008)
        return x, spatial, score


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class FIDInceptionA(nn.Module):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels: int, pool_features: int) -> None:
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # NOTE: Tensorflow's average pool does not use the padded zero's in its average calculation
        # https://github.com/mseitzer/pytorch-fid/blob/b9c18118d082cbd263c1b8963fc4221dc1cbb659/src/pytorch_fid/inception.py#L236-L240
        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(nn.Module):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels: int, channels_7x7: int) -> None:
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # NOTE: Tensorflow's average pool does not use the padded zero's in its average calculation
        # https://github.com/mseitzer/pytorch-fid/blob/b9c18118d082cbd263c1b8963fc4221dc1cbb659/src/pytorch_fid/inception.py#L266-L270
        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE(nn.Module):
    """InceptionE block patched for FID computation"""
    def __init__(self, in_channels: int, pool_type: str = "avg") -> None:
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        if pool_type == "avg":
            # NOTE: Tensorflow's average pool does not use the padded zero's in its average calculation
            # https://github.com/mseitzer/pytorch-fid/blob/b9c18118d082cbd263c1b8963fc4221dc1cbb659/src/pytorch_fid/inception.py#L301-L305
            self.branch_pool_func = lambda x: F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        elif pool_type == "max":
            # NOTE: Patch: The FID Inception model uses max pooling instead of average pooling.
            # This is likely an error in this specific Inception implementation, as other Inception models use average pooling here
            # (which matches the description in the paper).
            # https://github.com/mseitzer/pytorch-fid/blob/b9c18118d082cbd263c1b8963fc4221dc1cbb659/src/pytorch_fid/inception.py#L336-L340
            self.branch_pool_func = lambda x: F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"Unsupported pool_type `{pool_type}`")
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = self.branch_pool_func(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
