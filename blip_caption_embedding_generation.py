"""LAVIS BLIP Captioning.

Initial slow version without custom question example. Not all extensions tested if working.

References:
    - https://github.com/salesforce/LAVIS
    - https://github.com/salesforce/LAVIS/blob/main/examples/blip_image_captioning.ipynb
    - https://en.wikipedia.org/wiki/Image_file_format
    - https://blog.ml6.eu/the-art-of-pooling-embeddings-c56575114cf8
    - https://huggingface.co/sentence-transformers/all-mpnet-base-v2
"""
import os
import argparse
import pathlib

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.functional import F
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from lavis.models import load_model


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def generate_caption(args: argparse.Namespace) -> None:
    args.dest = args.dest or args.src

    caption_dest_path = f'{args.dest}/caption'
    caption_embed_dest_path = f'{args.dest}/caption_embed'
    caption_token_embed_dest_path = f'{args.dest}/caption_token_embed'
    caption_token_mask_embed_dest_path = f'{args.dest}/caption_token_mask_embed'
    pathlib.Path(caption_dest_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(caption_embed_dest_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(caption_token_embed_dest_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(caption_token_mask_embed_dest_path).mkdir(parents=True, exist_ok=True)

    if args.embed_cap or args.cap_path:
        st_text_embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        st_text_embed_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    if args.ckpt:
        vis_processor = load_processor("blip_image_eval").build(image_size=args.img_size)
        model = load_model(
            name="blip_caption",
            model_type="large_coco",
            is_eval=True,
            device=args.device,
            checkpoint=args.ckpt,
        )
    else:
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip_caption", model_type="large_coco", is_eval=True, device=args.device
        )
        vis_processor = vis_processors['eval']

    ext_check = (".jpg", ".jpeg", ".png")
    if args.ext:
        ext_check += (".jfif", ".webp", ".bmp")

    if args.cap_path is not None:
        for root, dirs, files in os.walk(args.cap_path):
            for file in tqdm(files):
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), 'r') as f:
                        caption_list = [line.rstrip() for line in f]

                        encoded_input = st_text_embed_tokenizer(
                            caption_list,
                            padding=True,
                            truncation=True,
                            max_length=128,
                            return_tensors='pt'
                        )

                        # Compute token embeddings
                        with torch.no_grad():
                            caption_token_embeddings = st_text_embed_model(**encoded_input)

                        if args.embed_cap_token:
                            np.save(os.path.join(
                                caption_token_embed_dest_path,
                                f'token_{os.path.splitext(file)[0]}.npy'),
                                caption_token_embeddings.last_hidden_state
                            )

                            np.save(os.path.join(
                                caption_token_mask_embed_dest_path,
                                f'mask_{os.path.splitext(file)[0]}.npy'),
                                encoded_input['attention_mask']
                            )

                        if args.mean_pool:
                            caption_embeddings = mean_pooling(caption_token_embeddings, encoded_input['attention_mask'])
                            caption_embeddings = F.normalize(caption_embeddings, p=2, dim=1)
                        else:
                            caption_embeddings = caption_token_embeddings.pooler_output

                        np.save(os.path.join(
                            caption_embed_dest_path,
                            f'pooled_embed_{os.path.splitext(file)[0]}.npy'),
                            caption_embeddings
                        )

        return

    for root, dirs, files in os.walk(args.src):
        for file in tqdm(files):
            if file.endswith(ext_check):
                img_path = os.path.join(root, file)
                raw_image = Image.open(img_path).convert("RGB")
                image = vis_processor(raw_image).unsqueeze(0).to(args.device)

                if args.beam:
                    caption_list = model.generate(
                        {"image": image},
                        max_length=args.max_len,
                        min_length=args.min_len,
                        top_p=0.9,
                    )
                else:
                    caption_list = model.generate(
                        {"image": image},
                        use_nucleus_sampling=True,
                        num_captions=args.cap,
                        max_length=args.max_len,
                        min_length=args.min_len,
                        top_p=0.9,
                    )

                if args.embed_cap:
                    encoded_input = st_text_embed_tokenizer(
                        caption_list,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    )

                    # Compute token embeddings
                    with torch.no_grad():
                        caption_token_embeddings = st_text_embed_model(**encoded_input)

                    if args.embed_cap_token:
                        np.save(os.path.join(
                            caption_token_embed_dest_path,
                            f'token_{os.path.splitext(file)[0]}.npy'),
                            caption_token_embeddings.last_hidden_state
                        )

                        np.save(os.path.join(
                            caption_token_mask_embed_dest_path,
                            f'mask_{os.path.splitext(file)[0]}.npy'),
                            encoded_input['attention_mask']
                        )

                    if args.mean_pool:
                        caption_embeddings = mean_pooling(caption_token_embeddings, encoded_input['attention_mask'])
                        caption_embeddings = F.normalize(caption_embeddings, p=2, dim=1)
                    else:
                        caption_embeddings = caption_token_embeddings.pooler_output

                    np.save(os.path.join(
                        caption_embed_dest_path,
                        f'pooled_embed_{os.path.splitext(file)[0]}.npy'),
                        caption_embeddings
                    )

                output = '\n'.join(caption_list) + '\n'
                with open(os.path.join(caption_dest_path, f'{os.path.splitext(file)[0]}.txt'), 'w') as f:
                    f.write(output)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate captions from images.')
    parser.add_argument('-s', '--src', required=True, help="path to source images folder", type=pathlib.Path)
    parser.add_argument('-d', '--dest', help="path to output folder", type=pathlib.Path)
    parser.add_argument('--device', help="use cpu or cuda gpu", default='cuda', type=str)
    parser.add_argument('--ckpt', help="optional custom path to checkpoint", type=pathlib.Path)
    parser.add_argument('--img_size', metavar='--img-size', help="image size for processing", default=384, type=int)
    parser.add_argument('--max_len', metavar='--max-len', help="max output length", default=48, type=int)
    parser.add_argument('--min_len', metavar='--min-len', help="min output length", default=16, type=int)
    parser.add_argument('--cap', help="number of captions using neucleus sampling", default=1, type=int)
    parser.add_argument('--beam', help="generates single caption without sampling", action='store_true')
    parser.add_argument('--ext', help="extended mode supporting more image types", action='store_true')
    parser.add_argument('--cap_path', help="path to pre-generated captions for creating embeddings", type=pathlib.Path)
    parser.add_argument(
        '--embed_cap',
        help="use a sentence transformer model to generate embeddings from full caption text",
        action='store_true'
    )
    parser.add_argument(
        '--embed_cap_token',
        help="use a sentence transformer model to generate embeddings from caption text per token",
        action='store_true'
    )
    parser.add_argument(
        '--mean_pool',
        help='uses custom mean pooling for single embedding per sentence otherwise uses models own pool output',
        action='store_true'
    )
    args = parser.parse_args()
    generate_caption(args)


if __name__ == '__main__':
    main()
