from PIL import Image
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
from pathlib import Path
from collections import namedtuple

# from modules import modelloader
import sys
import re

blip_image_eval_size = 384
device = "cuda"

loaded_categories = []
re_topn = re.compile(r"\.top(\d+)\.")
Category = namedtuple("Category", ["name", "topn", "items"])


def rank(image_features, text_array, top_count=1, clip_model=None, dtype=None):
    import clip

    interrogate_clip_dict_limit = 80

    if interrogate_clip_dict_limit != 0:
        text_array = text_array[0 : int(interrogate_clip_dict_limit)]

    top_count = min(top_count, len(text_array))
    text_tokens = clip.tokenize([text for text in text_array], truncate=True).to(device)
    text_features = clip_model.encode_text(text_tokens).type(dtype)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = torch.zeros((1, len(text_array))).to(device)
    for i in range(image_features.shape[0]):
        similarity += (
            100.0 * image_features[i].unsqueeze(0) @ text_features.T
        ).softmax(dim=-1)
    similarity /= image_features.shape[0]

    top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
    return [
        (text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy() * 100))
        for i in range(top_count)
    ]


def download_default_clip_interrogate_categories(content_dir):
    print("Downloading CLIP categories...")

    tmpdir = f"{content_dir}_tmp"
    category_types = ["artists", "flavors", "mediums", "movements"]

    try:
        os.makedirs(tmpdir, exist_ok=True)
        for category_type in category_types:
            torch.hub.download_url_to_file(
                f"https://raw.githubusercontent.com/pharmapsychotic/clip-interrogator/main/clip_interrogator/data/{category_type}.txt",
                os.path.join(tmpdir, f"{category_type}.txt"),
            )
        os.rename(tmpdir, content_dir)

    except Exception as e:
        print(e, "downloading default CLIP interrogate categories")
    finally:
        if os.path.exists(tmpdir):
            os.removedirs(tmpdir)


def categories(content_dir="interrogate"):
    if not os.path.exists(content_dir):
        download_default_clip_interrogate_categories(content_dir)

    if os.path.exists(content_dir):
        skip_categories = []
        category_types = []
        for filename in Path(content_dir).glob('*.txt'):
            category_types.append(filename.stem)
            if filename.stem in skip_categories:
                continue
            m = re_topn.search(filename.stem)
            topn = 1 if m is None else int(m.group(1))
            with open(filename, "r", encoding="utf8") as file:
                lines = [x.strip() for x in file.readlines()]
            loaded_categories.append(
                Category(name=filename.stem, topn=topn, items=lines)
            )

    return loaded_categories


def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return

    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)

    for root, dirs, files in os.walk(path):
        for filename in files:
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext not in allowed_extensions:
                    continue

            yield os.path.join(root, filename)


def load_models(
    model_path: str,
    model_url: str = None,
    command_path: str = None,
    ext_filter=None,
    download_name=None,
    ext_blacklist=None,
) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    output = []

    try:
        places = []

        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(
                command_path, 'experiments/pretrained_models'
            )
            if os.path.exists(pretrained_path):
                print(f"Appending path: {pretrained_path}")
                places.append(pretrained_path)
            elif os.path.exists(command_path):
                places.append(command_path)

        places.append(model_path)

        for place in places:
            for full_path in walk_files(place, allowed_extensions=ext_filter):
                if os.path.islink(full_path) and not os.path.exists(full_path):
                    print(f"Skipping broken symlink: {full_path}")
                    continue
                if ext_blacklist is not None and any(
                    [full_path.endswith(x) for x in ext_blacklist]
                ):
                    continue
                if full_path not in output:
                    output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                from basicsr.utils.download_util import load_file_from_url

                dl = load_file_from_url(model_url, model_path, True, download_name)
                output.append(dl)
            else:
                output.append(model_url)

    except Exception as e:
        print(e)
        pass

    return output


def create_fake_fairscale():
    class FakeFairscale:
        def checkpoint_wrapper(self):
            pass

    sys.modules["fairscale.nn.checkpoint.checkpoint_activations"] = FakeFairscale


def load_blip_model():
    create_fake_fairscale()
    sys.path.append("./repositories/BLIP/")
    import models.blip

    # model_path = "/home/zzx/PythonProject/stable-diffusion-webui/models"
    # BLIP_path = "/home/zzx/PythonProject/stable-diffusion-webui/repositories/BLIP"
    model_path = "./weight"
    BLIP_path = "./"

    files = load_models(
        model_path=os.path.join(model_path, "BLIP"),
        model_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',
        ext_filter=[".pth"],
        download_name='model_base_caption_capfilt_large.pth',
    )
    print(files)

    blip_model = models.blip.blip_decoder(
        pretrained=files[0],
        image_size=blip_image_eval_size,
        vit='base',
        med_config=os.path.join(BLIP_path, "configs", "med_config.json"),
    )
    blip_model.eval()

    return blip_model


def load_clip_model():
    import clip

    running_on_cpu = False
    clip_model_name = 'ViT-L/14'

    if running_on_cpu:
        model, preprocess = clip.load(
            clip_model_name,
            device="cpu",
            download_root=None,
        )
    else:
        model, preprocess = clip.load(clip_model_name, download_root=None)

    model.eval()
    model = model.to("cuda")

    return model, preprocess


def generate_caption(pil_image, dtype, blip_model):
    gpu_image = (
        transforms.Compose(
            [
                transforms.Resize(
                    (blip_image_eval_size, blip_image_eval_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )(pil_image)
        .unsqueeze(0)
        .type(dtype)
        .to(device)
    )

    with torch.no_grad():
        caption = blip_model.generate(
            gpu_image, sample=False, num_beams=1, min_length=24, max_length=48
        )

    return caption[0]


def load():
    blip_model = load_blip_model()

    blip_model = blip_model.to(device)

    clip_model, clip_preprocess = load_clip_model()
    dtype = next(clip_model.parameters()).dtype
    return blip_model, clip_model, clip_preprocess, dtype


def img2text(pil_image, clip_model, clip_preprocess, blip_model, dtype):
    interrogate_return_ranks = False
    caption = generate_caption(pil_image, dtype=dtype, blip_model=blip_model)
    res = caption
    clip_image = clip_preprocess(pil_image).unsqueeze(0).type(dtype).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(clip_image).type(dtype)

        image_features /= image_features.norm(dim=-1, keepdim=True)

        for name, topn, items in categories():
            matches = rank(
                image_features,
                text_array=items,
                top_count=topn,
                clip_model=clip_model,
                dtype=dtype,
            )
            for match, score in matches:
                if interrogate_return_ranks:
                    res += f", ({match}:{score/100:.3f})"
                else:
                    res += f", {match}"
    return res


if __name__ == "__main__":
    blip_model, clip_model, clip_preprocess, dtype = load()
    import glob
    img_path = "./test.jpg"
    pil_image = Image.open(img_path).convert("RGB")
    res = img2text(pil_image, clip_model, clip_preprocess, blip_model, dtype)
    print(res)