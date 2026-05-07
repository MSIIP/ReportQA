import argparse
import base64
import io
import json
import os
from PIL import Image
from tqdm import tqdm

import nibabel as nib
import numpy as np
from openai import OpenAI


args = argparse.ArgumentParser()
args.add_argument("--input_path", type=str, default=None)
args.add_argument("--output_dir", type=str, default=None)
args.add_argument("--model", type=str, default=None)
args.add_argument("--base_url", type=str, default=None)
args.add_argument("--api_key", type=str, default=None)
args.add_argument("--num_slices", type=int, default=3)
args.add_argument("--window_center", type=float, default=None)
args.add_argument("--window_width", type=float, default=None)
args = args.parse_args()

client = OpenAI(
    base_url=args.base_url,
    api_key=args.api_key,
)
system_prompt=""


def pil_to_data_url(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def build_user_content(
    item,
    num_slices=3,
    window_center=None,
    window_width=None,
):
    """
    Build OpenAI multimodal message content:
    [
        {"type": "image_url", "image_url": {"url": "..."}},
        {"type": "image_url", "image_url": {"url": "..."}},
        {"type": "text", "text": "..."}
    ]
    """
    content = []

    for nii_path in item["videos"]:
        if not os.path.exists(nii_path):
            print(f"[Warning] NII file not found: {nii_path}")
            continue

        nii = nib.load(nii_path)
        image3d = nii.get_fdata()  # (X, Y, Z) == (W, H, T)

        if window_center is not None and window_width is not None:
            hu_min = window_center - window_width // 2
            hu_max = window_center + window_width // 2
            image3d = np.clip(image3d, hu_min, hu_max)
            image3d = (image3d - hu_min) / (hu_max - hu_min) * 255

        depth = image3d.shape[2]
        # indices = np.linspace(0, depth - 1, num_slices, dtype=int)
        indices = np.linspace(0, depth - 1, num_slices + 2, dtype=int)[1:-1]

        slices = []
        for idx in indices:
            slice = image3d[:, :, idx]
            img = Image.fromarray(slice).convert("RGB")
            slices.append(img)

        for slice in slices:
            data_url = pil_to_data_url(slice)
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

    prompt = item["messages"][-2]["content"]
    content.append({
        "type": "text",
        "text": prompt.replace("<video>", "").strip(),
    })
    return content


if __name__ == "__main__":
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, item in tqdm(enumerate(data), total=len(data)):
        if os.path.exists(os.path.join(args.output_dir, f"{idx}.json")):
            continue

        try:
            user_content = build_user_content(
                item,
                num_slices=args.num_slices,
                window_center=args.window_center,
                window_width=args.window_width
            )
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                # temperature=0.2,
                # max_tokens=1024,
            )
            output_text = response.choices[0].message.content
        except Exception as e:
            print(f"[Error] Failed to get response for item {item['id']}: {e}")
            continue

        # item["messages"][-1]["content"] = output_text
        item["predict"] = output_text
        print(f"Predict:\n{output_text}\n")

        with open(os.path.join(args.output_dir, f"{idx}.json"), "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=4)

        # if idx > 3:
        #     break
