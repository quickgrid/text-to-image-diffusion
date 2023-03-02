# text-to-image-diffusion
Experimental (working!) custom implementation of conditional and unconditional diffusion for testing new methods. Based on [this repo](https://github.com/quickgrid/pytorch-diffusion). 

### Recommended 
- `accumulation_iters` * `accumulation_batch_size` atleast 32.

### Todo
- Todo.

### Codes
| Name | Description |
| --- | --- |
| [old_codes](https://github.com/quickgrid/text-to-image-diffusion/tree/main/old_codes) | Previous codes in the repo. |
| `caption_embedding_generator.py` | Generates and saves text captions of images and pooled embedding, token embedding, mask of captions. |
| `basic_diffusion.py` | Modifies [ddpm example](https://github.com/quickgrid/pytorch-diffusion) and [old_codes](https://github.com/quickgrid/text-to-image-diffusion/tree/main/old_codes) to add new features that works properly. |
| `custom_diffusion.py` | Testing new architectures, methods that may or may not work. |

### References

- Todo.
