import torch
import torch.nn as nn



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x





class GlobalGeometricEnhancer(nn.Module):
    def __init__(self, embed_dim=128, depth=4, num_heads=12, drop_path_rate=0.1):
        super().__init__()
        self.trans_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate

        # pos embedding for each patch
        self.pos_embed = nn.Sequential(
            nn.Linear(3, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, self.trans_dim)
        )

        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(embed_dim=self.trans_dim, depth=self.depth,
                                         drop_path_rate=dpr, num_heads=self.num_heads, mlp_ratio=1)
        # layer norm
        self.norm = nn.LayerNorm(self.trans_dim)


    def forward(self, center_embedding, center_position):
        pos = center_position.permute(0, 2, 1)  # TO: B, N_center, Dim
        pos = self.pos_embed(pos)  # B, N_center, Dim

        x = center_embedding.permute(0, 2, 1)  # TO: B, N_center, 3
        x = self.blocks(x, pos)
        x = self.norm(x)

        x = x.permute(0, 2, 1)  # O: B, DIM, N_Center
        return x


if __name__ == '__main__':
    center_embedding = torch.rand(2, 512, 128).cuda()  ## B, Dim, N_points  (if Dim=512 num_head=8, Dim=320 num_head=5)
    center_position = torch.rand(2, 3, 128).cuda()
    net = GlobalGeometricEnhancer(embed_dim=512, depth=1, num_heads=8).to(center_embedding.device)
    out = net(center_embedding, center_position)

    # for name, parameters in net.named_parameters():  # 打印出每一层的参数的大小
    #     print(name, ':', parameters.size())
    #
    # for param_tensor in net.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    #     print(param_tensor, '\t', net.state_dict()[param_tensor].size())

    print("Total number of paramerters in networks is {} M ".format(sum(x.numel() / 1e6 for x in net.parameters())))






