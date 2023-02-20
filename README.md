# text-to-image-diffusion
Experimental (working!) custom implementation of conditional and unconditional diffusion for testing new methods. Based on [this repo](https://github.com/quickgrid/pytorch-diffusion). Not properly tested.


### Codes
| Name | Description |
| --- | --- |
| `blip_caption_embedding_generation.py` | Generates and saves text captions of images, text sentence pooled embedding, also per token embedding of caption. |
| `ddpm.py` | Modifies [ddpm example](https://github.com/quickgrid/pytorch-diffusion) to support text conditioning. |
