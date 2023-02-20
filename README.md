# text-to-image-diffusion
Experimental (working!) custom implementation of conditional and unconditional diffusion for testing new methods. Based on [this repo](https://github.com/quickgrid/pytorch-diffusion). 

Tested only on 2000 image celebahq dataset for small amount of time. Not sure if actually working.

### Recommended 
- `accumulation_iters` * `batch_size` atleast 16.

### Codes
| Name | Description |
| --- | --- |
| `blip_caption_embedding_generation.py` | Generates and saves text captions of images, text sentence pooled embedding, also per token embedding of caption. |
| `ddpm.py` | Modifies [ddpm example](https://github.com/quickgrid/pytorch-diffusion) to support text conditioning. |
