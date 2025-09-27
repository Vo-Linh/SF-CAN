# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional datasets

from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .uda_dataset import UDADataset
from .loveda import LoveDADataset, SMLoveDADataset
from .smda_dataset import SMDADataset
from .isprs import ISPRSDataset, SSISPRSDataset
from .potsdam import PotsdamDataset, SSPotsdamDataset

__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'UDADataset',
    'LoveDADataset',
    'SMLoveDADataset',
    'SMDADataset',
    'ISPRSDataset',
    'SSISPRSDataset',
    'PotsdamDataset',
    'SSPotsdamDataset',
]
