import argparse
import json
import os


args = argparse.ArgumentParser()
args.add_argument("--results_path", type=str, default=None)
args.add_argument("--results_abnormal_path", type=str, default=None)
args = args.parse_args()
os.makedirs(os.path.dirname(args.results_abnormal_path), exist_ok=True)


if __name__ == "__main__":
    with open(args.results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    results_abnormal = {}
    results_normal = {}
    for outer_report_id, item in results.items():
        if "/abnormal/" in outer_report_id:
            results_abnormal[outer_report_id] = item
        else:
            results_normal[outer_report_id] = item

    print(f"Total cases: {len(results)}")
    print(f"Abnormal cases: {len(results_abnormal)}")
    print(f"Normal cases: {len(results_normal)}")

    with open(args.results_abnormal_path, "w", encoding="utf-8") as f:
        json.dump(results_abnormal, f, indent=4, ensure_ascii=False)
