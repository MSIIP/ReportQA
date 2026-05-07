import argparse
import json
import os
from tqdm import tqdm


args = argparse.ArgumentParser()
args.add_argument("--output_dir", type=str, default=None)
args.add_argument("--output_path", type=str, default=None)
args = args.parse_args()


if __name__ == "__main__":
    file_path_list = [file_path for file_path in os.listdir(args.output_dir) if file_path.endswith(".json")]
    file_path_list = sorted(file_path_list, key=lambda x: int(x.split(".")[0]))

    results = []
    for file_path in tqdm(file_path_list):
        print(f"Processing {file_path}...")
        with open(os.path.join(args.output_dir, file_path), "r", encoding="utf-8") as f:
            result = json.load(f)
            results.append(result)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
