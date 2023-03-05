"""Experimental diffusion based text to image generation.

References:
    - https://github.com/quickgrid/paper-implementations/blob/main/pytorch/vision_transformer/vit.py
    - https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    - https://github.com/lucidrains/memory-efficient-attention-pytorch
    - Self Attention Paper, https://arxiv.org/pdf/1706.03762.pdf
    - ViT Paper, https://arxiv.org/pdf/2010.11929.pdf
"""
import torch
from torch import nn
from einops.layers.torch import Rearrange
from memory_efficient_attention_pytorch.flash_attention import FlashAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            mlp_hidden_dim: int = 1024,
            num_head: int = 8,
            per_dim_head: int = 64,
            dropout: float = 0.1,
    ):
        """Runs embedded patches through transformer encoder. Figure 1 of ViT paper.
        """
        super(TransformerEncoderBlock, self).__init__()

        self.layer_norm = nn.LayerNorm([dim])
        self.attn = FlashAttention(dim=dim, heads=num_head, dim_head=per_dim_head)
        self.mlp = nn.Sequential(
            nn.LayerNorm([dim]),
            nn.Linear(in_features=dim, out_features=mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_hidden_dim, out_features=dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.attn(x=self.layer_norm(x), context=context, mask=mask) + x
        return self.mlp(x) + x


class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks: int, **kwargs):
        """Receives same keyword arguments from `TransformerEncoderBlock`.
        """
        super(TransformerEncoder, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(TransformerEncoderBlock(**kwargs))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        for transformer_block in self.blocks:
            x = transformer_block(x=x, context=context, mask=mask)
        return x


class ViT(nn.Module):
    def __init__(
            self,
            dim: int,
            image_size: int,
            patch_size: int,
            emb_dropout: int = 0.1,
            img_channels: int = 3,
            num_blocks: int = 2,
            **kwargs,
    ):
        """ViT implementation without class token. Assumes square input image tensor in shape of (b, c, h, w) and
        square patch size. Uses learned 1D positional encoding.
        """
        super(ViT, self).__init__()

        patch_tmp = image_size // patch_size
        self.num_patches = patch_tmp ** 2
        patch_dim = img_channels * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_size, pw=patch_size),
            nn.LayerNorm([patch_dim]),
            nn.Linear(in_features=patch_dim, out_features=dim),
            nn.LayerNorm([dim]),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(p=emb_dropout)

        self.transformer_encoder = TransformerEncoder(
            dim=dim,
            num_blocks=num_blocks,
            **kwargs,
        )

        self.to_img_pixels = nn.Sequential(
            nn.Linear(in_features=dim, out_features=patch_dim),
            nn.LayerNorm([patch_dim]),
            Rearrange(
                'b (h w) (ph pw c) -> b c (h ph) (w pw)',
                ph=patch_size,
                pw=patch_size,
                c=img_channels,
                h=patch_tmp,
                w=patch_tmp,
            ),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.to_patch_embedding(x)
        print(self.pos_embedding.shape, x.shape)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer_encoder(x=x, context=context, mask=mask)
        return self.to_img_pixels(x)


v = torch.randn((4, 3, 256, 256)).cuda()
m = ViT(num_blocks=2, dim=768, image_size=256, patch_size=16).cuda()
out = m(v)
print(out.shape)

