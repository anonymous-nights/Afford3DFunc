import torch
from model.model import Afford3DModel
from torch.utils.data import DataLoader
from dataset.AffordanceNet import AffordNetDataset
from utils.eval import evaluation
from config.config import *


if __name__ == "__main__":

    model = Afford3DModel().cuda()
    model.load_state_dict(torch.load(checkpoint))

    val_set = AffordNetDataset(data_root, 'val')
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=8, drop_last=False)
    mIoU = evaluation(model, val_loader, zero_shot_affordance)
