# text-to-image-diffusion
Experimental custom implementation of conditional and unconditional diffusion for testing new methods. Based on [this repo](https://github.com/quickgrid/pytorch-diffusion). 

### Recommended 
- `accumulation_iters` * `accumulation_batch_size` atleast 32.

### Todo

Temporary merged todo for this, [vqcompress](https://github.com/quickgrid/vq-compress), [distillsd](https://github.com/quickgrid/distill-sd) repo.

- Train on vqgan autoencoding model indices using [vqcompress](https://github.com/quickgrid/vq-compress) to learn instead of lower dim encoded output. In this case indices as input will only be `64x64` instead of using the output `64x64x3` for vq-f8 512x512 resolution. Check if a simpler network can learn these integer better than vae outputs. This may need less memory in attention, generated result quality may be slightly worse with vqgan than autoencoder kl pretrained decoder.  
- Train on ldm autoencoding models encoded lower dim output `64x64x4` for kl-f8. 
- Add xformers, custom flash attention copied from.
- Train only on indices or output on lower dim and use vae decoder only after certain steps to check results.
- Improve unet, attention, transformer implementation, add learned positional embeddings.
- Reduce gpu training and inference memory.
- Add quanization aware training, try post training quanization, dynamic quantization. 
- See if feasible to use image decomposition methods from [vqcompress](https://github.com/quickgrid/vq-compress) repo to degrade images instead of `T`th step gaussian noise. Add image quality assesment results for non-trained image decomposition and compression methods.
- Add ddpm for in code to use but use ddim as default for training and results.
- Check how to add other noise, sampling methods to code.
- Try to add own super resolution following papers without using ldm vae decoder.
- Try to add inpainting to code.
- Split ldm model vae, text encoder, unet and convert them to onnx to check performance.
- Split pretrained ldm stable diffusion unet in half and distill half of unet then rest half using [distillsd](https://github.com/quickgrid/distill-sd). 
- Split vae autoencoder encoder, decoder and load only the necessary ones for task.
- Check dwt watermark code.
- Try to add accelerate library. 
- Check if useful fake tensors and deferred module init from [here](https://pytorch.org/torchdistx/latest/fake_tensor_and_deferred_init.html).


### Codes
| Name | Description |
| --- | --- |
| [old_codes](https://github.com/quickgrid/text-to-image-diffusion/tree/main/old_codes) | Previous codes in the repo. |
| `caption_embedding_generator.py` | Generates and saves text captions of images and pooled embedding, token embedding, mask of captions. |
| `basic_diffusion.py` | Modifies [ddpm example](https://github.com/quickgrid/pytorch-diffusion) and [old_codes](https://github.com/quickgrid/text-to-image-diffusion/tree/main/old_codes) to add new features that works properly. |
| `exp_diffusion.py` | Testing new architectures, methods that may or may not work. |

### References

- https://github.com/lucidrains/memory-efficient-attention-pytorch
