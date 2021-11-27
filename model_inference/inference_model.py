import os
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_metric_learning import losses
from transformers import ViTFeatureExtractor, ViTModel

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()