"""PyTorch Module and ModuleGenerator."""

from src.modules.base_generator import GeneratorAbstract, ModuleGenerator
from src.modules.bottleneck import Bottleneck, BottleneckGenerator
from src.modules.conv import Conv, ConvGenerator, FixedConvGenerator
from src.modules.dwconv import DWConv, DWConvGenerator
from src.modules.flatten import FlattenGenerator
from src.modules.invertedresidualv2 import (InvertedResidualv2,
                                            InvertedResidualv2Generator)
from src.modules.invertedresidualv3 import (InvertedResidualv3,
                                            InvertedResidualv3Generator)
from src.modules.linear import Linear, LinearGenerator
from src.modules.poolings import (AvgPoolGenerator, GlobalAvgPool,
                                  GlobalAvgPoolGenerator, MaxPoolGenerator)

from src.modules.scaledDotProductAttention import ScaledDotProductAttentionGenerator,CoAtConvGenerator
from src.modules.poolings import MaxPool1dGenerator                                 
from src.modules.mbconv import MBConvGenerator
from src.modules.swin import SwinTransformerGenerator
from src.modules.efficientnet import EfficientNetGenerator
from src.modules.mbv3 import MBv3SmallGenerator
from src.modules.shufflenetv2 import ShuffleNetV2Generator
from src.modules.shufflenetv2 import ShuffleNetV1Generator

__all__ = [
    "ModuleGenerator",
    "GeneratorAbstract",
    "Bottleneck",
    "Conv",
    "DWConv",
    "Linear",
    "GlobalAvgPool",
    "InvertedResidualv2",
    "InvertedResidualv3",
    "BottleneckGenerator",
    "FixedConvGenerator",
    "ConvGenerator",
    "LinearGenerator",
    "DWConvGenerator",
    "FlattenGenerator",
    "MaxPoolGenerator",
    "AvgPoolGenerator",
    "GlobalAvgPoolGenerator",
    "InvertedResidualv2Generator",
    "InvertedResidualv3Generator",

    "MBConvGenerator",
    "ScaledDotProductAttentionGenerator",
    "CoAtConvGenerator",
    "MaxPool1dGenerator"
    "SwinTransformerGenerator",
    "EfficientNetGenerator",
    'MBv3SmallGenerator',
    'ShuffleNetV2Generator',
    "ShuffleNetV2Generator",
]
