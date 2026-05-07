import argparse
import json
import os
from tqdm import tqdm

from openai import OpenAI


args = argparse.ArgumentParser()
args.add_argument("--input_path", type=str, default=None)
args.add_argument("--output_dir", type=str, default=None)
args.add_argument("--model", type=str, default=None)
args.add_argument("--base_url", type=str, default=None)
args.add_argument("--api_key", type=str, default=None)
args = args.parse_args()

client = OpenAI(
    base_url=args.base_url,
    api_key=args.api_key,
)
system_prompt=""


def build_user_content(item):
    """
    Build OpenAI multimodal message content:
    [
        {"type": "image_url", "image_url": {"url": "..."}},
        {"type": "image_url", "image_url": {"url": "..."}},
        {"type": "text", "text": "..."}
    ]
    """
    content = []
    prompt = item["messages"][-2]["content"]
    content.append({
        "type": "text",
        "text": prompt.replace("<image>", "").replace("<video>", "").strip(),
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
            user_content = build_user_content(item)
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
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
