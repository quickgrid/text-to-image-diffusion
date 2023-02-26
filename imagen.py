"""Imagen implementation.

References:
    - Demo implementation, https://github.com/quickgrid/paper-implementations/tree/main/pytorch/imagen.
    - Imagen paper, https://arxiv.org/abs/2205.11487.
    - Self Attention paper, https://arxiv.org/abs/1706.03762.
    - Vision Transformers paper, https://arxiv.org/pdf/2010.11929.pdf.
"""
import math
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.functional import F
from tqdm import tqdm

from memory_efficient_attention import Attention


class EfficientUNetResNetBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_groups: int = 8,
    ):
        """Efficient UNet implementation from Figure A.27.

        Input channels are split in `num_groups` with each group having `in_channels / num_groups` channels. Groupnorm,
        https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html.

        SiLU is used in place of Swish as both are same functions,
        https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html.
        """
        super(EfficientUNetResNetBlock, self).__init__()

        self.main_path = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.skip_path = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main_path(x) + self.skip_path(x)


class EfficientUNetDBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            cond_embed_dim: int,
            num_resnet_blocks: int,
            contextual_text_embed_dim: int = None,
            stride: Tuple[int, int] = None,
            use_text_conditioning: bool = False,
            attn_resolution: int = 32,
    ):
        """Implementation of Efficient UNet DBlock as shown in Figure A.28. If stide is provided downsamples
        input tensor by the amount.

        Embedding layers are used to bring feature map shape to expected embedding dimension from different input
        dimensions.

        In paper first conv in DBlock is optional with strided downsampling. Conv block is kept and only down samples
        when stride is provided else keeps same shape in h, w.

        Args:
            out_channels: Current block expected output channels.
            num_resnet_blocks: Number of sequential resnet blocks in dblock between CombineEmbs and SelfAttention.
            cond_embed_dim: Conditinal embeddings dimension like time, class, pooled text embeddings in
                shape of (batch_size, embed_dim).
            contextual_text_embed_dim: Per token text embedding from pretrain language model in
                shape (batch_size, seq_size, embed_dim).
            stride: With (1, 1) output has same h, w as input with shape of (batch_size, out_channel, h, w).
                With stride of (2, 2) downsamples tensor as (batch_size, out_channel, h / 2, w / 2).
            use_text_conditioning: Cross Self Attention with text embedding is used for text to image.
        """
        super(EfficientUNetDBlock, self).__init__()
        self.use_text_conditioning = use_text_conditioning
        self.use_conv = True if stride is not None else False

        self.initial_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), stride=stride
        )

        self.conditional_embedding_layer = nn.Sequential(
            nn.Linear(in_features=cond_embed_dim, out_features=out_channels)
        )

        self.resnet_blocks = nn.Sequential()
        for _ in range(num_resnet_blocks):
            self.resnet_blocks.append(
                EfficientUNetResNetBlock(in_channels=out_channels, out_channels=out_channels)
            )

        if use_text_conditioning:
            self.contextual_text_embedding_layer = nn.Sequential(
                nn.Linear(in_features=contextual_text_embed_dim, out_features=out_channels)
            )

        self.mem_efficient_attn = Attention(
            dim=out_channels,
            dim_head=attn_resolution,  # dimension per head
            heads=8,  # number of attention heads
            causal=True,  # autoregressive or not
            memory_efficient=True,
            # whether to use memory efficient attention (can be turned off to test against normal attention)
            q_bucket_size=1024,  # bucket size along queries dimension
            k_bucket_size=2048  # bucket size along key / values dimension
        )

    def forward(
            self,
            x: torch.Tensor,
            conditional_embedding: torch.Tensor,
            contextual_text_embedding: torch.Tensor = None,
            contextual_text_mask_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """DBlock, initial conv (optional) -> combine embs -> resnet blocks -> self attention (optional).

        Expected conditional_embedding shape (batch, 1, 1, cond_embed_dim), which is passed through embedding layer.
        Embedding layer converts cond_embed_dim to out_channels to match initial conv output shape. The output shape
        is (batch, 1, 1, out_channels).

        The conditional embedding, `cond_embed` is reshaped to add with input feature map x. Channel first
        format used in the code, but channel last can be used by reshaping x. Out channel feature map is replicated
        along height and width per pixel. If shape of output of initial conv is (batch, out_channels, hw, hw) then
        `cond_embed` is converted to,
        (batch, 1, 1, out_channels) -> (batch, out_channels, 1, 1) -> (batch, out_channels, hw, hw).

        Attention is only used if defined for that layer.

        Input `conditional_embedding` and `contextual_text_embedding` shape in channel dimension do not need to be
        same as they are projected to expected shape `output_channels` with embedding layers.

        Args:
            x: Previous DBlock output.
            conditional_embedding: Time, class, pooled text embedding. Example shape, (batch, embedding_dim).
            contextual_text_embedding: Contextual text embedding from pretrained model like T5. Example shape,
                (batch, token_size, embedding_dim).
            contextual_text_mask_embedding: Mask generated by model in shape (batch, token_size).
        """
        x = self.initial_conv(x) if self.use_conv else x
        cond_embed = self.conditional_embedding_layer(conditional_embedding)
        cond_embed = cond_embed.view(
            cond_embed.shape[0], cond_embed.shape[1], 1, 1
        ).repeat(1, 1, x.shape[2], x.shape[3])
        x = x + cond_embed
        x = self.resnet_blocks(x)

        if self.use_text_conditioning:
            context_text_embed = self.contextual_text_embedding_layer(contextual_text_embedding)
            b, c, h, w = x.shape
            x = x.view(b, c, h * w).permute(0, 2, 1)
            x = self.mem_efficient_attn(x=x, context=context_text_embed, mask=contextual_text_mask_embedding)
            x = x.permute(0, 2, 1).view(b, c, h, w)

        return x


class EfficientUNetUBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            cond_embed_dim: int,
            num_resnet_blocks: int,
            stride: Tuple[int, int] = None,
            use_attention: bool = False,
            attn_resolution: int = 32,
    ):
        """Implementation of Efficient UNet UBlock as shown in Figure A.29.

        Rather than not having conv block when stride is not provided it is kept. It upsamples if stride is provided
        else keeps the same shape in spatial dimension.
        """
        super(EfficientUNetUBlock, self).__init__()
        self.use_attention = use_attention
        self.use_conv = True if stride is not None else False

        self.conditional_embedding_layer = nn.Sequential(
            nn.Linear(in_features=cond_embed_dim, out_features=out_channels)
        )

        self.input_embedding_layer = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)
        )

        self.resnet_blocks = nn.Sequential()
        for _ in range(num_resnet_blocks):
            self.resnet_blocks.append(
                EfficientUNetResNetBlock(in_channels=out_channels, out_channels=out_channels)
            )

        if use_attention:
            self.mem_efficient_attn = Attention(
                dim=out_channels,
                dim_head=attn_resolution,  # dimension per head
                heads=8,  # number of attention heads
                causal=True,  # autoregressive or not
                memory_efficient=True,
                # whether to use memory efficient attention (can be turned off to test against normal attention)
                q_bucket_size=1024,  # bucket size along queries dimension
                k_bucket_size=2048  # bucket size along key / values dimension
            )

        self.last_conv_upsampler = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
            ),
            nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True),
        )

    def forward(
            self,
            x: torch.Tensor,
            conditional_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Ublock, combine embs -> resnet blocks -> self attention (optional) -> last conv (optional).

        Args:
            x: Previous UBlock output.
            conditional_embedding: Time, class, pooled Text embeddings.
        """
        cond_embed = self.conditional_embedding_layer(conditional_embedding)
        cond_embed = cond_embed.view(
            cond_embed.shape[0], cond_embed.shape[1], 1, 1
        ).repeat(1, 1, x.shape[2], x.shape[3])
        x = self.input_embedding_layer(x)
        x = x + cond_embed
        x = self.resnet_blocks(x)

        if self.use_attention:
            b, c, h, w = x.shape
            x = x.view(b, c, h * w).permute(0, 2, 1)
            x = self.mem_efficient_attn(x)
            x = x.permute(0, 2, 1).view(b, c, h, w)

        if self.use_conv:
            x = self.last_conv_upsampler(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            dropout: float = 0.1,
            max_len: int = 1000,
    ):
        """Section 3.5 of attention is all you need paper.

        Extended slicing method is used to fill even and odd position of sin, cos with increment of 2.
        Ex, `[sin, cos, sin, cos, sin, cos]` for `embedding_dim = 6`.

        `max_len` is equivalent to number of noise steps or patches. `embedding_dim` must same as image
        embedding dimension of the model.

        Args:
            embedding_dim: `d_model` in given positional encoding formula.
            dropout: Dropout amount.
            max_len: Number of embeddings to generate. Here, equivalent to total noise steps.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(name='pos_encoding', tensor=pos_encoding, persistent=False)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        """Get precalculated positional embedding at timestep t. Outputs same as video implementation
        code but embeddings are in [sin, cos, sin, cos] format instead of [sin, sin, cos, cos] in that code.
        Also batch dimension is added to final output.
        """
        positional_encoding = self.pos_encoding[t].squeeze(1)
        return self.dropout(positional_encoding)


class Diffusion:
    def __init__(
            self,
            device: str,
            img_size: int,
            noise_steps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
    ):
        self.device = device
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        # Section 2, equation 4 and near explation for alpha, alpha hat, beta.
        self.beta = self.linear_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Section 3.2, algorithm 1 formula implementation. Generate values early reuse later.
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

        # Section 3.2, equation 2 precalculation values.
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.std_beta = torch.sqrt(self.beta)

        # Clean up unnecessary values.
        del self.alpha
        del self.alpha_hat

    def linear_noise_schedule(self) -> torch.Tensor:
        """Same amount of noise is applied each step. Weakness is near end steps image is so noisy it is hard make
        out information. So noise removal is also very small amount, so it takes more steps to generate clear image.
        """
        return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps, device=self.device)

    def q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Section 3.2, algorithm 1 formula implementation. Forward process, defined by `q`.

        Found in section 2. `q` gradually adds gaussian noise according to variance schedule. Also,
        can be seen on figure 2.
        """
        sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Random timestep for each sample in a batch. Timesteps selected from [1, noise_steps].
        """
        return torch.randint(low=1, high=self.noise_steps, size=(batch_size,), device=self.device)

    def p_sample(
            self,
            eps_model: nn.Module,
            n: int,
            scale_factor: int = 2,
            conditional_embedding: torch.Tensor = None,
            contextual_text_embedding: torch.Tensor = None,
            contextual_text_mask_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """Implementation of algorithm 2 sampling. Reverse process, defined by `p` in section 2. Short
         formula is defined in equation 11 of section 3.2.

        From noise generates image step by step. From noise_steps, (noise_steps - 1), ...., 2, 1.
        Here, alpha = 1 - beta. So, beta = 1 - alpha.

        Sample noise from normal distribution of timestep t > 1, else noise is 0. Before returning values
        are clamped to [-1, 1] and converted to pixel values [0, 255].

        Args:
            contextual_text_mask_embedding:
            contextual_text_embedding:
            conditional_embedding: Pooled embedding (same shape per string) generated by language model.
            scale_factor: Scales the output image by the factor.
            eps_model: Noise prediction model. `eps_theta(x_t, t)` in paper. Theta is the model parameters.
            n: Number of samples to process.

        Returns:
            Generated denoised image.
        """
        print(f'Sampling {n} images....')

        eps_model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = torch.ones(n, dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1, 1)
                beta_t = self.beta[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1, 1, 1)

                random_noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = ((1 / sqrt_alpha_t) *
                     (x - ((beta_t / sqrt_one_minus_alpha_hat_t) *
                           eps_model(
                               x=x,
                               timestep=t,
                               conditional_embedding=conditional_embedding,
                               contextual_text_embedding=contextual_text_embedding,
                               contextual_text_mask_embedding=contextual_text_mask_embedding,

                           )))
                     ) + (epsilon_t * random_noise)

        eps_model.train()

        x = ((x.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
        x = F.interpolate(input=x, scale_factor=scale_factor, mode='nearest-exact')
        return x


class EfficientUNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            cond_embed_dim: int = 768,
            base_channel_dim: int = 32,
            num_resnet_blocks: Union[Tuple[int, ...], int] = None,
            channel_mults: Tuple[int, ...] = None,
            contextual_text_embed_dim: int = None,
            use_text_conditioning: Tuple[bool, ...] = False,
            use_attention: Tuple[bool, ...] = True,
            attn_resolution: Tuple[int, ...] = 32,
            noise_steps: int = 1000,
    ):
        """UNet implementation for 64 x 64 image as defined in Section F.1 and efficient UNet architecture for
         64 -> 256 upsampling as shown in Figure A.30.

        Ellipsis used for variable number of U and D blocks. The number of D and U blocks depend on the number
        of `channel_mults`.

        Parameter of the current UNet model does not depend on the image resolution.

        TODO: In cascade diffusion may need to concat low res image with noisy image which should result in 6 channels.

        Args:
            in_channels: Input image tensor channels.
            cond_embed_dim: Timestep or text embedding output dimension.
            base_channel_dim: Base value for multiplying with channel_mults for U or D blocks of UNet.
            num_resnet_blocks: Number of resnet blocks in each of the U or D blocks of UNet.
            channel_mults: Multiplier values for each of the U or D blocks in UNet.
        """
        super(EfficientUNet, self).__init__()
        if channel_mults is None:
            channel_mults = (1, 2, 3, 4)
        if num_resnet_blocks is None:
            num_resnet_blocks = 3

        if isinstance(num_resnet_blocks, int):
            num_resnet_blocks = (num_resnet_blocks,) * len(channel_mults)

        assert len(channel_mults) == len(num_resnet_blocks), 'channel_mults and num_resnet_blocks should be same shape.'

        mutliplied_channels = np.array(channel_mults) * base_channel_dim
        mutliplied_channels_len = len(mutliplied_channels)
        mutliplied_channels_reversed = np.flip(mutliplied_channels)
        num_resnet_blocks_reversed = np.flip(num_resnet_blocks)

        self.initial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mutliplied_channels[0],
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        self.dblocks = nn.ModuleList()
        for idx, num_channels in enumerate(mutliplied_channels[:-1]):
            self.dblocks.append(
                EfficientUNetDBlock(
                    in_channels=mutliplied_channels[idx],
                    out_channels=mutliplied_channels[idx + 1],
                    cond_embed_dim=cond_embed_dim,
                    num_resnet_blocks=num_resnet_blocks[idx],
                    stride=(2, 2),
                    contextual_text_embed_dim=contextual_text_embed_dim,
                    use_text_conditioning=use_text_conditioning[idx],
                    attn_resolution=attn_resolution[idx]
                )
            )

        self.ublocks = nn.ModuleList()
        self.ublocks.append(
            EfficientUNetUBlock(
                in_channels=mutliplied_channels_reversed[0],
                out_channels=mutliplied_channels_reversed[1],
                cond_embed_dim=cond_embed_dim,
                num_resnet_blocks=num_resnet_blocks_reversed[1],
                stride=(2, 2),
                use_attention=use_attention[0],
                attn_resolution=attn_resolution[0],
            )
        )
        for idx in range(1, mutliplied_channels_len - 1, 1):
            self.ublocks.append(
                EfficientUNetUBlock(
                    in_channels=mutliplied_channels_reversed[idx] * 2,
                    out_channels=mutliplied_channels_reversed[idx + 1],
                    cond_embed_dim=cond_embed_dim,
                    num_resnet_blocks=num_resnet_blocks_reversed[idx],
                    stride=(2, 2),
                    use_attention=use_attention[idx],
                    attn_resolution=attn_resolution[idx]
                )
            )

        self.image_projection = nn.Conv2d(
            in_channels=channel_mults[0] * base_channel_dim, out_channels=3, kernel_size=(3, 3), padding=(1, 1)
        )

        self.pos_encoding = PositionalEncoding(embedding_dim=cond_embed_dim, max_len=noise_steps)

    def forward(
            self,
            x: torch.Tensor,
            timestep: torch.LongTensor,
            conditional_embedding: torch.Tensor,
            contextual_text_embedding: torch.Tensor = None,
            contextual_text_mask_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """Efficient UNet forward for given number of unet blocks.

        As shown in Figure A.30 the last unet dblock and first unet block in the middle do not have skip connection.
        """
        x = self.initial_conv(x)

        timestep = self.pos_encoding(timestep)
        conditional_embedding += timestep

        x_skip_outputs = []
        for dblock in self.dblocks:
            x = dblock(x, conditional_embedding, contextual_text_embedding, contextual_text_mask_embedding)
            x_skip_outputs.append(x)

        x_skip_outputs.pop()
        x = self.ublocks[0](x=x, conditional_embedding=conditional_embedding)

        for ublock in self.ublocks[1:]:
            x = torch.cat((x, x_skip_outputs.pop()), dim=1)
            x = ublock(x=x, conditional_embedding=conditional_embedding)

        x = self.image_projection(x)
        return x
