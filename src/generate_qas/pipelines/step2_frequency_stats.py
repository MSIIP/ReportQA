import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


EMBEDDING_CLUSTER_DEFAULT_FIELDS = ("sign", "disease", "anatomy")
TASK_CLUSTER_FIELDS = {
    "finding": {"sign", "anatomy"},
    "diagnosis": {"disease", "anatomy"},
    "both": set(EMBEDDING_CLUSTER_DEFAULT_FIELDS),
}


def get_counter_name(content_type, key):
    if key == "clinical_entity":
        return "sign" if content_type == "finding" else "disease"
    if key == "location":
        return "anatomy"
    return f"attribute_{key}"


def count_feature_frequencies(jsonl_path, content_type):
    field_counters = defaultdict(Counter)

    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON解析失败:")
                print(f"文件: {jsonl_path}")
                print(f"行号: {lineno}")
                print(f"错误: {e}")
                print(f"内容(前200字符): {line[:200]}")
                print(f"内容(后200字符): {line[-200:]}")
                continue   # 👉 跳过错误行，继续处理

            features = data.get("extracted_features", [])
            for feat in features:
                for key, value in feat.items():
                    if key == "evidence_span":
                        continue
                    if value is None or value == "":
                        continue

                    counter_name = get_counter_name(content_type, key)
                    field_counters[counter_name][value] += 1

    return field_counters


def merge_counters(counter_list):
    """
    将多个 field_counters 合并，key 相同的 Counter 累加
    """
    merged = defaultdict(Counter)
    for counter in counter_list:
        for key, value in counter.items():
            merged[key].update(value)
    return merged


def resolve_cluster_fields(task):
    try:
        return set(TASK_CLUSTER_FIELDS[task])
    except KeyError as exc:
        raise ValueError(f"不支持的 task: {task}") from exc


def l2_normalize(vector):
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def cosine_similarity(vec1, vec2):
    return sum(x * y for x, y in zip(vec1, vec2))


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


class TransformerEmbedder:
    def __init__(self, model_name, device="cpu"):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "启用 embedding 聚类需要安装 torch 和 transformers。"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def encode(self, texts, batch_size=32):
        embeddings = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                embeddings.extend(pooled.cpu().tolist())
        return embeddings


def choose_standard_term(counter, terms):
    return sorted(terms, key=lambda term: (-counter[term], term))[0]


def cluster_counter_by_embeddings(counter, embedder, similarity_threshold):
    terms = sorted(counter.keys())
    if len(terms) <= 1:
        clusters = []
        if terms:
            term = terms[0]
            clusters.append(
                {
                    "standard_term": term,
                    "total_frequency": counter[term],
                    "members": {term: counter[term]},
                    "size": 1,
                }
            )
        return Counter(counter), {term: term for term in terms}, clusters

    embeddings = [l2_normalize(vec) for vec in embedder.encode(terms)]
    union_find = UnionFind(len(terms))

    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if cosine_similarity(embeddings[i], embeddings[j]) >= similarity_threshold:
                union_find.union(i, j)

    grouped_terms = defaultdict(list)
    for index, term in enumerate(terms):
        grouped_terms[union_find.find(index)].append(term)

    normalized_counter = Counter()
    term_mapping = {}
    clusters = []

    for member_terms in grouped_terms.values():
        standard_term = choose_standard_term(counter, member_terms)
        members = dict(
            sorted(
                ((term, counter[term]) for term in member_terms),
                key=lambda item: (-item[1], item[0]),
            )
        )
        total_frequency = sum(members.values())
        normalized_counter[standard_term] = total_frequency
        for term in member_terms:
            term_mapping[term] = standard_term
        clusters.append(
            {
                "standard_term": standard_term,
                "total_frequency": total_frequency,
                "members": members,
                "size": len(member_terms),
            }
        )

    clusters.sort(key=lambda item: (-item["total_frequency"], item["standard_term"]))
    return normalized_counter, term_mapping, clusters


def should_cluster_field(field_name, cluster_fields, cluster_attributes):
    if field_name in cluster_fields:
        return True
    return cluster_attributes and field_name.startswith("attribute_")


def normalize_counters_with_embeddings(field_counters, embedder, cluster_fields, similarity_threshold, cluster_attributes=True):
    normalized = defaultdict(Counter)
    cluster_details = {}

    for field_name, counter in field_counters.items():
        if not should_cluster_field(field_name, cluster_fields, cluster_attributes):
            normalized[field_name] = Counter(counter)
            continue

        normalized_counter, term_mapping, clusters = cluster_counter_by_embeddings(
            counter,
            embedder,
            similarity_threshold,
        )
        normalized[field_name] = normalized_counter
        cluster_details[field_name] = {
            "similarity_threshold": similarity_threshold,
            "term_mapping": dict(sorted(term_mapping.items(), key=lambda item: item[0])),
            "clusters": clusters,
        }

    return normalized, cluster_details


def serialize_counters(field_counters):
    serialized = {}
    for key, value in field_counters.items():
        sorted_items = sorted(value.items(), key=lambda item: (-item[1], item[0]))
        if key.startswith("attribute_"):
            attribute_field = key[len("attribute_") :]
            serialized.setdefault("attribute", {})[attribute_field] = dict(sorted_items)
        else:
            serialized[key] = dict(sorted_items)
    return serialized


def main():
    parser = argparse.ArgumentParser(description="统计特征频数并保存为 JSON")
    parser.add_argument("--finding_file", type=str, help="finding JSONL 文件路径")
    parser.add_argument("--diagnosis_file", type=str, help="diagnosis JSONL 文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="统计结果 JSON 文件路径")
    parser.add_argument("--task", type=str, choices=["finding", "diagnosis", "both"], default="both")
    parser.add_argument(
        "--enable_embedding_cluster",
        action="store_true",
        help="对指定字段进行 embedding 向量化并按余弦相似度聚类归一化",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-m3",
        help="embedding 模型名或本地模型目录，例如 BAAI/bge-m3 或 ClinicalBERT 路径",
    )
    parser.add_argument(
        "--cluster_similarity_threshold",
        type=float,
        default=0.9,
        help="实体自动聚类的余弦相似度阈值",
    )
    parser.add_argument(
        "--embedding_device",
        type=str,
        default="cpu",
        help="embedding 推理设备，例如 cpu、cuda",
    )
    args = parser.parse_args()

    counters_list = []
    if args.task in ["finding", "both"]:
        if not args.finding_file:
            raise ValueError("当 --task 为 finding 或 both 时，必须提供 --finding_file")
        counters_list.append(count_feature_frequencies(args.finding_file, "finding"))

    if args.task in ["diagnosis", "both"]:
        if not args.diagnosis_file:
            raise ValueError("当 --task 为 diagnosis 或 both 时，必须提供 --diagnosis_file")
        counters_list.append(count_feature_frequencies(args.diagnosis_file, "diagnosis"))

    merged_counters = merge_counters(counters_list)
    normalized_counters = merged_counters
    normalization_info = None

    if args.enable_embedding_cluster:
        embedder = TransformerEmbedder(args.embedding_model, device=args.embedding_device)
        cluster_fields = resolve_cluster_fields(args.task)
        normalized_counters, cluster_details = normalize_counters_with_embeddings(
            merged_counters,
            embedder,
            cluster_fields=cluster_fields,
            similarity_threshold=args.cluster_similarity_threshold,
            cluster_attributes=True,
        )
        normalization_info = {
            "enabled": True,
            "method": "embedding_cosine_clustering",
            "embedding_model": args.embedding_model,
            "embedding_device": args.embedding_device,
            "task": args.task,
            "cluster_fields": sorted(cluster_fields),
            "cluster_attributes": True,
            "similarity_threshold": args.cluster_similarity_threshold,
            "cluster_details": cluster_details,
        }
    else:
        normalization_info = {
            "enabled": False,
        }

    output_data = {
        "task": args.task,
        "normalization": normalization_info,
        "stats": serialize_counters(normalized_counters),
        "stats_raw": serialize_counters(merged_counters),
    }

    with Path(args.output_file).open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"频数统计完成，结果保存至 {args.output_file}")


if __name__ == "__main__":
    main()
