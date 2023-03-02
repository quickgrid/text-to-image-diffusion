# text-to-image-diffusion
Experimental (working!) custom implementation of conditional and unconditional diffusion for testing new methods. Based on [this repo](https://github.com/quickgrid/pytorch-diffusion). 

`ddpm_basic.py` tested only on 2000 image celebahq dataset for small amount of time and somewhat works.

### Recommended 
- `accumulation_iters` * `accumulation_batch_size` atleast 32.

### Todo
- Fix gif generation code.
- Fix unconditional training.

### Codes
| Name | Description |
| --- | --- |
| `blip_caption_embedding_generation.py` | Generates and saves text captions of images and pooled embedding, token embedding, mask of captions. |
| `ddpm_basic.py` | Modifies [ddpm example](https://github.com/quickgrid/pytorch-diffusion) to support pooled text conditioning. |
| `ddpm_full.py` | Modifies [ddpm example](https://github.com/quickgrid/pytorch-diffusion) to support pooled text, token conditioning. |
| `experiment.py` | Modifies [imagen example](https://github.com/quickgrid/paper-implementations/tree/main/pytorch/imagen). **Does not work yet.** |
