import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
import numpy as np
from model.fte import compose_method
from model.pointnet import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from model.attention import GlobalGeometricEnhancer


class ClassEncoder(nn.Module):
    def __init__(self):
        super(ClassEncoder, self).__init__()
        self.device = torch.device('cuda')
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    def forward(self, classes):
        tokens = clip.tokenize(classes).to(self.device)
        text_features = self.clip_model.encode_text(tokens).to(self.device).permute(1, 0).float()
        return text_features

cls_encoder = ClassEncoder()


class Afford3DModel(nn.Module):
    def __init__(self, normal_channel=False, full_version_flag=True):
        super(Afford3DModel, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        return_group_idx = True

        self.full_version_flag = full_version_flag

        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128],
                                             3 + additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                                             return_group_idx)
        if self.full_version_flag:
            self.gem1 = GlobalGeometricEnhancer(embed_dim=320, depth=1, num_heads=5)
            self.gem2 = GlobalGeometricEnhancer(embed_dim=512, depth=1, num_heads=8)

        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128],
                                             128 + 128 + 64, [[128, 128, 256], [128, 196, 256]],
                                             return_group_idx)

        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1024, mlp=[128])
        self.fp2 = PointNetFeaturePropagation(in_channel=448, mlp=[128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134 + additional_channel, mlp=[128])

        self.dim_changer0 = nn.Conv1d(128, 512, 1)
        self.dim_changer1 = nn.Conv1d(128, 512, 1)
        self.dim_changer2 = nn.Conv1d(128, 512, 1)

        self.logit_scale0 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.logit_scale0_ = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale1_ = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2_ = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, xyz, affordance, isTrain=True):
        with torch.no_grad():
            if isTrain:
                affordance = compose_method(affordance, method=5)
            text_features = cls_encoder(affordance)

        # Set Abstraction layers
        xyz = xyz.contiguous()
        B, C, N = xyz.shape
        l0_idxes = [torch.arange(N, dtype=torch.long).to(xyz.device).repeat(B, 1).unsqueeze(-1)]
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_xyz = xyz
            l0_points = xyz

        l1_xyz, l1_points, l1_idxes = self.sa1(l0_xyz, l0_points)  ## l1_points: B, DIM, N (16, 320, 512)
        if self.full_version_flag:
            l1_points = self.gem1(l1_points, l1_xyz) + l1_points
        l2_xyz, l2_points, l2_idxes = self.sa2(l1_xyz, l1_points)  ## l2_points: B, DIM, N (16, 512, 128)
        if self.full_version_flag:
            l2_points = self.gem2(l2_points, l2_xyz) + l2_points  # 1203/1204
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  ## l3_points:B, DIM, N (16, 512, 1)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # B, 128D, 128n
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)   # B, DIM, Np , B, 128D, 512n
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)  # [B, 128D, 2048]

        ## dim changer
        l0_points = self.dim_changer0(l0_points).permute(0, 2, 1).float()  ## B, 2048, 512D
        l1_points = self.dim_changer1(l1_points).permute(0, 2, 1).float()  ## B, 512 512D
        l2_points = self.dim_changer2(l2_points).permute(0, 2, 1).float()  ## B, 128, 512D

        # Text Norm
        text_features_ = text_features / torch.norm(text_features, dim=0, keepdim=True).repeat(512, 1)

        # Point Norm
        l0_points_ = l0_points / torch.norm(l0_points, dim=2, keepdim=True).repeat(1, 1, 512)
        l1_points_ = l1_points / torch.norm(l1_points, dim=2, keepdim=True).repeat(1, 1, 512)
        l2_points_ = l2_points / torch.norm(l2_points, dim=2, keepdim=True).repeat(1, 1, 512)

        # Co-relation
        points = [l0_points_, l1_points_, l2_points_]
        idxes = [l0_idxes, l1_idxes, l2_idxes]
        logits_scale_list = [self.logit_scale0, self.logit_scale1, self.logit_scale2]
        logits_scale_list_ = [self.logit_scale0_, self.logit_scale1_, self.logit_scale2_]
        v2ts = []
        t2vs = []
        for i in range(len(points)):
            v2t = F.log_softmax(logits_scale_list[i] * (points[i] @ text_features_).permute(0,2,1), dim=1)
            t2v = F.softmax(logits_scale_list_[i] * (points[i].reshape(-1, 512) @ text_features_), dim=0)
            v2ts.append(v2t)
            t2vs.append(t2v)

        return v2ts, t2vs, idxes





if __name__ == '__main__':

    net = Afford3DModel()
    print("Total number of paramerters in networks is {} M ".format(sum(x.numel() / 1e6 for x in net.parameters())))










