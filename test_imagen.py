"""Tests for Imagen code.
"""

import torch

from imagen import (
    EfficientUNetDBlock,
    EfficientUNetUBlock,
    EfficientUNet,
)


class TestImagen:
    def __init__(self, device: str = 'cuda', print_arch: bool = True):
        self.device = device
        self.print_arch = print_arch

    def test_dblock(self):
        # x: Tensor, shape[seq_len, batch_size, embedding_dim]
        embedding_dim = 768
        num_channels = 32
        hw = 64
        b = 4
        x = torch.randn((b, num_channels, hw, hw), device=self.device)
        print(f'in: {x.shape}')
        cond_embedding = torch.randn((b, embedding_dim), device=self.device)
        context_embedding = torch.randn((b, 13, embedding_dim), device=self.device)
        context_mask_embedding = torch.ones((b, 13), dtype=torch.bool, device=self.device)

        t_enc = EfficientUNetDBlock(
            in_channels=num_channels,
            out_channels=128,
            cond_embed_dim=768,
            num_resnet_blocks=2,
            stride=(1, 1),
        ).to(self.device)
        out = t_enc(x, cond_embedding)
        print(f'out: {out.shape}')

        t_enc = EfficientUNetDBlock(
            in_channels=num_channels,
            out_channels=128,
            cond_embed_dim=768,
            num_resnet_blocks=2,
            stride=(1, 1),
            contextual_text_embed_dim=768,
            use_text_conditioning=True,
        ).to(self.device)
        out = t_enc(x, cond_embedding, context_embedding)
        print(f'out: {out.shape}')

        t_enc = EfficientUNetDBlock(
            in_channels=num_channels,
            out_channels=128,
            cond_embed_dim=768,
            num_resnet_blocks=2,
            stride=(1, 1),
            contextual_text_embed_dim=768,
            use_text_conditioning=True,
        ).to(self.device)
        out = t_enc(x, cond_embedding, context_embedding, context_mask_embedding)
        print(f'out: {out.shape}')

        t_enc = EfficientUNetDBlock(
            in_channels=num_channels,
            out_channels=128,
            cond_embed_dim=768,
            num_resnet_blocks=2,
            stride=(2, 2),
            contextual_text_embed_dim=768,
            use_text_conditioning=True,
        ).to(self.device)
        out = t_enc(x, cond_embedding, context_embedding, context_mask_embedding)
        print(f'out: {out.shape}')

    def test_ublock(self):
        # x: Tensor, shape[seq_len, batch_size, embedding_dim]
        embedding_dim = 768
        num_channels = 32
        hw = 64
        b = 4
        x = torch.randn((b, num_channels, hw, hw), device=self.device)
        print(f'in: {x.shape}')
        cond_embedding = torch.randn((b, embedding_dim), device=self.device)

        t_enc = EfficientUNetUBlock(
            in_channels=num_channels,
            out_channels=128,
            cond_embed_dim=768,
            num_resnet_blocks=2,
            stride=(1, 1),
            use_attention=True,
        ).to(self.device)
        out = t_enc(x, cond_embedding)
        print(f'out: {out.shape}')

        t_enc = EfficientUNetUBlock(
            in_channels=num_channels,
            out_channels=128,
            cond_embed_dim=768,
            num_resnet_blocks=2,
            stride=(2, 2),
            use_attention=True,
        ).to(self.device)
        out = t_enc(x, cond_embedding)
        print(f'out: {out.shape}')

    def test_efficient_unet(self):
        # x: Tensor, shape[seq_len, batch_size, embedding_dim]
        noise_steps = 500
        embedding_dim = 768
        num_channels = 3
        hw = 256
        b = 4
        x = torch.randn((b, num_channels, hw, hw), device=self.device)
        print(f'in: {x.shape}')
        cond_embedding = torch.randn((b, embedding_dim), device=self.device)
        context_embedding = torch.randn((b, 13, embedding_dim), device=self.device)
        context_mask_embedding = torch.ones((b, 13), dtype=torch.bool, device=self.device)

        timestep = torch.randint(low=1, high=noise_steps, size=(b,), device=self.device)

        t_enc = EfficientUNet(
            in_channels=num_channels,
            cond_embed_dim=embedding_dim,
            base_channel_dim=64,
            num_resnet_blocks=(2, 1, 1),
            channel_mults=(2, 1, 2),
            contextual_text_embed_dim=embedding_dim,
            use_text_conditioning=(False, True, True),
            use_attention=(False, True, True),
            attn_resolution=(32, 16, 8),
        ).to(self.device)

        out = t_enc(
            x=x,
            timestep=timestep,
            conditional_embedding=cond_embedding,
            contextual_text_embedding=context_embedding,
            contextual_text_mask_embedding=context_mask_embedding,
        )
        print(f'out: {out.shape}')


if __name__ == '__main__':
    tester = TestImagen(print_arch=False)

    # tester.test_dblock()
    # tester.test_ublock()
    tester.test_efficient_unet()
