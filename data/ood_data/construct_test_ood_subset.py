import re
import json
import math
import pickle
import random
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class OODSubsetConfig:
    # Dataset information
    dataset_name: str = "CMU-MOSI"

    # Official SDK test split.
    test_path: str = "/root/autodl-tmp/data/mosi_test.pkl"

    # Output directory.
    output_dir: str = "/root/autodl-tmp/data/ood_splits"

    # OOD construction setting:
    #   "binary_np" : negative / positive
    #   "binary_nn" : negative / non-negative
    #   "seven"     : seven-class sentiment intensity
    objective_task: str = "binary_np"

    # Sentiment-label field.
    # For binary_np or binary_nn, "classification_labels" is typically used.
    # For seven-class construction, use a field containing the original
    # sentiment scores or seven-class labels.
    label_field: str = "classification_labels"

    # Label scheme:
    #   "classification_3" : labels are 0, 1, 2
    #   "sentiment_score"  : labels are continuous scores in [-3, 3]
    #   "classification_7" : labels are seven discrete classes
    label_scheme: str = "classification_3"

    # Minimum sample-level word occurrence for vocabulary construction.
    # Since CMU-MOSI test split is relatively small, this threshold should
    # usually be lower than that used for the full dataset.
    min_word_count: int = 5

    # Number of OOD videos selected from the official test split.
    ood_video_count: int = 12

    # Target distribution difference.
    # This value should be reported together with the generated OOD subset.
    target_delta: float = 0.50

    # Simulated annealing parameters.
    outer_iterations: int = 800
    swaps_per_iteration: int = 200
    initial_temperature: float = 0.5
    temperature_decay: float = 0.99

    # Number of video groups exchanged in each perturbation step.
    max_swap_groups: int = 1

    # Random seed for reproducibility.
    seed: int = 42

    # Optional output prefix.
    output_prefix: Optional[str] = None


CONFIG = OODSubsetConfig()


# ============================================================
# 2. Basic I/O utilities
# ============================================================

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# 3. Text preprocessing
# ============================================================

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "you're", "you've", "you'll", "you'd", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "she's", "her", "hers", "herself", "it", "it's", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "that'll", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "don't", "should", "should've", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn",
    "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn",
    "hadn't", "hasn", "hasn't", "haven", "haven't", "isn",
    "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't",
    "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't",
    "wasn", "wasn't", "weren", "weren't", "won", "won't",
    "wouldn", "wouldn't"
}


def normalize_text_to_tokens(text: Any) -> List[str]:
    """
    Normalize raw text into a list of tokens.

    Apostrophes are preserved so that contractions remain valid tokens.
    """
    if isinstance(text, (list, tuple, np.ndarray)):
        text = " ".join(str(x) for x in text)
    else:
        text = str(text)

    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)

    tokens = []
    for token in text.split():
        token = token.strip("'")
        if not token:
            continue
        if token in STOP_WORDS:
            continue
        tokens.append(token)

    return tokens


def get_video_id(sample_id: Any) -> str:
    """
    Extract the video identifier from a segment identifier.

    Example:
        "-6rXp3zJ3kc$_$9" -> "-6rXp3zJ3kc"
    """
    sample_id = str(sample_id)
    if "$_$" in sample_id:
        return sample_id.split("$_$")[0]
    return sample_id


# ============================================================
# 4. Test split loading and subsetting
# ============================================================

def get_test_split_object(data: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Obtain the test split dictionary from an input pickle object.

    Returns:
        split_data  : dictionary containing fields such as raw_text, id, labels
        wrapper_key : None if the input itself is the split dictionary;
                      otherwise the key that stores the test split
    """
    if isinstance(data, dict) and "raw_text" in data:
        return data, None

    if isinstance(data, dict) and "test" in data and isinstance(data["test"], dict):
        if "raw_text" in data["test"]:
            return data["test"], "test"

    raise KeyError(
        "The input file must either be a split dictionary containing 'raw_text' "
        "or a dataset dictionary containing a 'test' split."
    )


def infer_split_length(split_data: Dict[str, Any]) -> int:
    if "raw_text" in split_data:
        return len(split_data["raw_text"])

    for value in split_data.values():
        try:
            return len(value)
        except Exception:
            continue

    raise ValueError("Unable to infer the number of samples.")


def subset_value(value: Any, indices: np.ndarray, split_length: int) -> Any:
    """
    Select a subset from a split-level field when its first dimension
    matches the number of samples.
    """
    indices = np.asarray(indices, dtype=np.int64)

    if isinstance(value, np.ndarray):
        if value.shape[0] == split_length:
            return value[indices]
        return value

    if isinstance(value, list):
        if len(value) == split_length:
            return [value[int(i)] for i in indices]
        return value

    if isinstance(value, tuple):
        if len(value) == split_length:
            return tuple(value[int(i)] for i in indices)
        return value

    try:
        if len(value) == split_length:
            return [value[int(i)] for i in indices]
    except Exception:
        pass

    return value


def subset_split_data(split_data: Dict[str, Any], indices: np.ndarray) -> Dict[str, Any]:
    """
    Select a subset of samples from the official test split.
    """
    split_length = infer_split_length(split_data)

    output = {}
    for key, value in split_data.items():
        output[key] = subset_value(value, indices, split_length)

    return output


def restore_output_format(
    original_data: Any,
    subset_data: Dict[str, Any],
    wrapper_key: Optional[str]
) -> Any:
    """
    Preserve the input pickle structure in the output file.
    """
    if wrapper_key is None:
        return subset_data

    output = dict(original_data)
    output[wrapper_key] = subset_data
    return output


# ============================================================
# 5. Sentiment-label mapping
# ============================================================

def get_num_classes(cfg: OODSubsetConfig) -> int:
    if cfg.objective_task in {"binary_np", "binary_nn"}:
        return 2
    if cfg.objective_task == "seven":
        return 7
    raise ValueError(
        "objective_task must be one of {'binary_np', 'binary_nn', 'seven'}."
    )


def map_label_to_category(
    label: float,
    cfg: OODSubsetConfig
) -> Optional[int]:
    """
    Map a sentiment label to the category used by the OOD objective.

    Returns:
        category index, or None if the sample is excluded from the
        word-distribution objective.
    """
    y = float(label)

    if cfg.objective_task == "binary_np":
        if cfg.label_scheme == "classification_3":
            if y < 1:
                return 0
            if y > 1:
                return 1
            return None

        if cfg.label_scheme == "sentiment_score":
            if y < 0:
                return 0
            if y > 0:
                return 1
            return None

        raise ValueError("binary_np requires classification_3 or sentiment_score labels.")

    if cfg.objective_task == "binary_nn":
        if cfg.label_scheme == "classification_3":
            return 0 if y < 1 else 1

        if cfg.label_scheme == "sentiment_score":
            return 0 if y < 0 else 1

        raise ValueError("binary_nn requires classification_3 or sentiment_score labels.")

    if cfg.objective_task == "seven":
        if cfg.label_scheme == "sentiment_score":
            category = int(np.clip(np.round(y), -3, 3)) + 3
            return category

        if cfg.label_scheme == "classification_7":
            if 0 <= y <= 6:
                return int(y)
            if -3 <= y <= 3:
                return int(y) + 3

        raise ValueError(
            "seven-class construction requires sentiment_score or classification_7 labels."
        )

    raise ValueError(f"Unsupported objective_task: {cfg.objective_task}")


def label_count(labels: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


# ============================================================
# 6. Word-occurrence statistics
# ============================================================

def build_vocabulary(
    token_lists: List[List[str]],
    labels: np.ndarray,
    cfg: OODSubsetConfig
) -> List[str]:
    """
    Build the vocabulary using sample-level word occurrence.

    A word is counted at most once within each sample.
    """
    counter = Counter()

    for tokens, label in zip(token_lists, labels):
        category = map_label_to_category(label, cfg)
        if category is None:
            continue
        counter.update(set(tokens))

    vocab = sorted([w for w, c in counter.items() if c > cfg.min_word_count])
    return vocab


def build_sample_word_class_counts(
    token_lists: List[List[str]],
    labels: np.ndarray,
    vocab: List[str],
    cfg: OODSubsetConfig
) -> np.ndarray:
    """
    Build a sample-level word-category occurrence tensor.

    Shape:
        [num_samples, vocab_size, num_classes]
    """
    word2idx = {word: i for i, word in enumerate(vocab)}
    num_samples = len(token_lists)
    vocab_size = len(vocab)
    num_classes = get_num_classes(cfg)

    counts = np.zeros((num_samples, vocab_size, num_classes), dtype=np.float64)

    for i, (tokens, label) in enumerate(zip(token_lists, labels)):
        category = map_label_to_category(label, cfg)
        if category is None:
            continue

        for word in set(tokens):
            j = word2idx.get(word)
            if j is not None:
                counts[i, j, category] = 1.0

    return counts


def compute_phi(word_class_counts: np.ndarray) -> np.ndarray:
    """
    Compute word distributions over sentiment categories.

    phi[w, c] = count(w, c) / sum_c count(w, c)
    """
    denominator = word_class_counts.sum(axis=1, keepdims=True)

    phi = np.divide(
        word_class_counts,
        denominator,
        out=np.zeros_like(word_class_counts, dtype=np.float64),
        where=(denominator > 0)
    )

    return phi


def objective_value(
    reference_counts: np.ndarray,
    ood_counts: np.ndarray,
    target_delta: np.ndarray
) -> float:
    """
    Compute the simulated annealing objective.

    V = || |phi_ref - phi_ood| - |phi_delta| ||_1
    """
    phi_ref = compute_phi(reference_counts)
    phi_ood = compute_phi(ood_counts)

    actual_delta = np.abs(phi_ref - phi_ood)
    target_delta = np.abs(target_delta)

    return float(np.abs(actual_delta - target_delta).sum())


# ============================================================
# 7. Video-level grouping
# ============================================================

def build_video_groups(sample_ids: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Group sample indices by video identifier.
    """
    group_to_indices = defaultdict(list)

    for i, sample_id in enumerate(sample_ids):
        video_id = get_video_id(sample_id)
        group_to_indices[video_id].append(i)

    video_ids = sorted(group_to_indices.keys())
    return video_ids, dict(group_to_indices)


def build_video_count_tensor(
    sample_counts: np.ndarray,
    video_ids: List[str],
    group_to_indices: Dict[str, List[int]]
) -> np.ndarray:
    """
    Aggregate sample-level counts into video-level counts.

    Shape:
        [num_videos, vocab_size, num_classes]
    """
    video_counts = []

    for video_id in video_ids:
        indices = group_to_indices[video_id]
        video_counts.append(sample_counts[indices].sum(axis=0))

    return np.stack(video_counts, axis=0)


def video_indices_to_sample_indices(
    selected_video_indices: np.ndarray,
    video_ids: List[str],
    group_to_indices: Dict[str, List[int]]
) -> np.ndarray:
    """
    Convert selected video indices into sample indices.
    """
    sample_indices = []

    for video_idx in selected_video_indices:
        video_id = video_ids[int(video_idx)]
        sample_indices.extend(group_to_indices[video_id])

    return np.asarray(sorted(sample_indices), dtype=np.int64)


# ============================================================
# 8. Simulated annealing
# ============================================================

def initialize_ood_mask(
    num_videos: int,
    ood_video_count: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Randomly initialize the OOD video mask.
    """
    if not (0 < ood_video_count < num_videos):
        raise ValueError(
            f"ood_video_count must be in (0, {num_videos}), got {ood_video_count}."
        )

    mask = np.zeros(num_videos, dtype=bool)
    selected = rng.choice(num_videos, size=ood_video_count, replace=False)
    mask[selected] = True
    return mask


def propose_video_swap(
    ood_mask: np.ndarray,
    max_swap_groups: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a candidate perturbation by exchanging video groups
    between the selected OOD subset and the reference subset.
    """
    ood_indices = np.flatnonzero(ood_mask)
    reference_indices = np.flatnonzero(~ood_mask)

    max_k = min(max_swap_groups, len(ood_indices), len(reference_indices))
    k = int(rng.integers(1, max_k + 1))

    remove_from_ood = rng.choice(ood_indices, size=k, replace=False)
    add_to_ood = rng.choice(reference_indices, size=k, replace=False)

    return remove_from_ood, add_to_ood


def should_accept(
    current_value: float,
    candidate_value: float,
    temperature: float,
    rng: np.random.Generator
) -> bool:
    """
    Metropolis acceptance criterion for minimizing the objective.
    """
    if candidate_value <= current_value:
        return True

    if temperature <= 0:
        return False

    probability = math.exp((current_value - candidate_value) / temperature)
    probability = min(1.0, probability)

    return rng.random() < probability


def run_simulated_annealing(
    video_counts: np.ndarray,
    cfg: OODSubsetConfig
) -> Dict[str, Any]:
    """
    Search for the OOD video subset within the official test split.
    """
    rng = np.random.default_rng(cfg.seed)

    num_videos = video_counts.shape[0]
    vocab_size = video_counts.shape[1]
    num_classes = video_counts.shape[2]

    target_delta = np.full(
        (vocab_size, num_classes),
        cfg.target_delta,
        dtype=np.float64
    )

    total_counts = video_counts.sum(axis=0)

    current_ood_mask = initialize_ood_mask(
        num_videos=num_videos,
        ood_video_count=cfg.ood_video_count,
        rng=rng
    )

    current_ood_counts = video_counts[current_ood_mask].sum(axis=0)
    current_reference_counts = total_counts - current_ood_counts

    current_value = objective_value(
        reference_counts=current_reference_counts,
        ood_counts=current_ood_counts,
        target_delta=target_delta
    )

    best_ood_mask = current_ood_mask.copy()
    best_ood_counts = current_ood_counts.copy()
    best_value = current_value

    temperature = cfg.initial_temperature
    history = []

    print("=" * 80)
    print("Simulated annealing for official test OOD subset")
    print(f"Number of test videos  : {num_videos}")
    print(f"Selected OOD videos    : {cfg.ood_video_count}")
    print(f"Vocabulary size        : {vocab_size}")
    print(f"Number of classes      : {num_classes}")
    print(f"Target delta           : {cfg.target_delta}")
    print(f"Outer iterations       : {cfg.outer_iterations}")
    print(f"Swaps per iteration    : {cfg.swaps_per_iteration}")
    print(f"Initial temperature    : {cfg.initial_temperature}")
    print(f"Temperature decay      : {cfg.temperature_decay}")
    print(f"Initial objective      : {current_value:.6f}")
    print("=" * 80)

    for outer in tqdm(range(cfg.outer_iterations), desc="Annealing"):
        accepted = 0
        improved = 0

        for _ in range(cfg.swaps_per_iteration):
            remove_from_ood, add_to_ood = propose_video_swap(
                ood_mask=current_ood_mask,
                max_swap_groups=cfg.max_swap_groups,
                rng=rng
            )

            add_counts = video_counts[add_to_ood].sum(axis=0)
            remove_counts = video_counts[remove_from_ood].sum(axis=0)

            candidate_ood_counts = current_ood_counts + add_counts - remove_counts
            candidate_reference_counts = total_counts - candidate_ood_counts

            candidate_value = objective_value(
                reference_counts=candidate_reference_counts,
                ood_counts=candidate_ood_counts,
                target_delta=target_delta
            )

            if should_accept(
                current_value=current_value,
                candidate_value=candidate_value,
                temperature=temperature,
                rng=rng
            ):
                current_ood_mask[remove_from_ood] = False
                current_ood_mask[add_to_ood] = True

                current_ood_counts = candidate_ood_counts
                current_value = candidate_value
                accepted += 1

                if current_value < best_value:
                    best_value = current_value
                    best_ood_mask = current_ood_mask.copy()
                    best_ood_counts = current_ood_counts.copy()
                    improved += 1

        record = {
            "iteration": outer + 1,
            "temperature": float(temperature),
            "current_value": float(current_value),
            "best_value": float(best_value),
            "accepted": int(accepted),
            "improved": int(improved),
            "accept_rate": float(accepted / max(cfg.swaps_per_iteration, 1))
        }
        history.append(record)

        if (outer + 1) % 50 == 0 or outer == 0:
            print(
                f"[{outer + 1:04d}] "
                f"current V={current_value:.6f}, "
                f"best V={best_value:.6f}, "
                f"T={temperature:.6f}, "
                f"accept={record['accept_rate']:.4f}"
            )

        temperature *= cfg.temperature_decay

    best_reference_counts = total_counts - best_ood_counts

    return {
        "best_ood_mask": best_ood_mask,
        "best_value": float(best_value),
        "best_reference_counts": best_reference_counts,
        "best_ood_counts": best_ood_counts,
        "target_delta": target_delta,
        "history": history,
    }


# ============================================================
# 9. Metadata utilities
# ============================================================

def class_count_for_objective(
    labels: np.ndarray,
    cfg: OODSubsetConfig
) -> Dict[str, int]:
    counts = defaultdict(int)

    for label in labels:
        category = map_label_to_category(label, cfg)
        if category is not None:
            counts[str(category)] += 1

    return dict(counts)


def compute_distribution_table(
    reference_counts: np.ndarray,
    ood_counts: np.ndarray,
    vocab: List[str],
    top_k: int = 30
) -> List[Dict[str, Any]]:
    """
    Summarize frequent words with large distribution shifts.
    """
    phi_ref = compute_phi(reference_counts)
    phi_ood = compute_phi(ood_counts)

    total_occurrence = reference_counts.sum(axis=1) + ood_counts.sum(axis=1)
    mean_abs_shift = np.abs(phi_ref - phi_ood).mean(axis=1)

    ranking_score = total_occurrence * mean_abs_shift
    top_indices = np.argsort(-ranking_score)[:top_k]

    table = []

    for idx in top_indices:
        idx = int(idx)

        table.append({
            "word": vocab[idx],
            "total_occurrence": int(total_occurrence[idx]),
            "reference_distribution": [
                float(x) for x in phi_ref[idx].tolist()
            ],
            "ood_distribution": [
                float(x) for x in phi_ood[idx].tolist()
            ],
            "mean_absolute_shift": float(mean_abs_shift[idx]),
        })

    return table


def summarize_subset(
    split_data: Dict[str, Any],
    labels: np.ndarray,
    sample_indices: np.ndarray,
    cfg: OODSubsetConfig
) -> Dict[str, Any]:
    """
    Summarize a selected subset.
    """
    subset_labels = labels[sample_indices]

    if "id" in split_data:
        ids = [str(split_data["id"][int(i)]) for i in sample_indices]
    else:
        ids = [str(int(i)) for i in sample_indices]

    video_ids = sorted(set(get_video_id(x) for x in ids))

    return {
        "num_samples": int(len(sample_indices)),
        "num_videos": int(len(video_ids)),
        "label_count": label_count(subset_labels),
        "objective_class_count": class_count_for_objective(subset_labels, cfg),
    }


# ============================================================
# 10. Main procedure
# ============================================================

def main(cfg: OODSubsetConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output_prefix is None:
        task_name = cfg.objective_task.replace("binary_", "")
        output_prefix = f"{cfg.dataset_name.lower().replace('-', '_')}_{task_name}_official_test_ood"
    else:
        output_prefix = cfg.output_prefix

    output_ood_path = output_dir / f"{output_prefix}.pkl"
    output_meta_path = output_dir / f"{output_prefix}_metadata.json"

    print("=" * 80)
    print("Loading official test split")
    print(f"Test path: {cfg.test_path}")
    print("=" * 80)

    original_data = load_pickle(cfg.test_path)
    test_data, wrapper_key = get_test_split_object(original_data)

    if "raw_text" not in test_data:
        raise KeyError("The field 'raw_text' is required.")

    if cfg.label_field not in test_data:
        raise KeyError(f"The label field '{cfg.label_field}' is required.")

    raw_texts = test_data["raw_text"]
    labels = np.asarray(test_data[cfg.label_field])

    if "id" in test_data:
        sample_ids = [str(x) for x in test_data["id"]]
    else:
        sample_ids = [str(i) for i in range(len(raw_texts))]

    if len(raw_texts) != len(labels):
        raise ValueError("The number of raw texts and labels does not match.")

    if len(sample_ids) != len(labels):
        raise ValueError("The number of sample identifiers and labels does not match.")

    token_lists = [normalize_text_to_tokens(x) for x in raw_texts]

    video_ids, group_to_indices = build_video_groups(sample_ids)
    num_videos = len(video_ids)

    print(f"Test samples : {len(labels)}")
    print(f"Test videos  : {num_videos}")
    print(f"Label count  : {label_count(labels)}")

    vocab = build_vocabulary(
        token_lists=token_lists,
        labels=labels,
        cfg=cfg
    )

    if len(vocab) == 0:
        raise ValueError(
            "The vocabulary is empty. Please reduce min_word_count or check the text field."
        )

    print("=" * 80)
    print("Vocabulary")
    print(f"Minimum word count : > {cfg.min_word_count}")
    print(f"Vocabulary size    : {len(vocab)}")
    print(f"First words        : {vocab[:20]}")
    print("=" * 80)

    sample_counts = build_sample_word_class_counts(
        token_lists=token_lists,
        labels=labels,
        vocab=vocab,
        cfg=cfg
    )

    video_counts = build_video_count_tensor(
        sample_counts=sample_counts,
        video_ids=video_ids,
        group_to_indices=group_to_indices
    )

    anneal_result = run_simulated_annealing(
        video_counts=video_counts,
        cfg=cfg
    )

    ood_video_mask = anneal_result["best_ood_mask"]
    ood_video_indices = np.flatnonzero(ood_video_mask)
    reference_video_indices = np.flatnonzero(~ood_video_mask)

    ood_sample_indices = video_indices_to_sample_indices(
        selected_video_indices=ood_video_indices,
        video_ids=video_ids,
        group_to_indices=group_to_indices
    )

    reference_sample_indices = video_indices_to_sample_indices(
        selected_video_indices=reference_video_indices,
        video_ids=video_ids,
        group_to_indices=group_to_indices
    )

    ood_subset_data = subset_split_data(test_data, ood_sample_indices)
    output_ood_data = restore_output_format(
        original_data=original_data,
        subset_data=ood_subset_data,
        wrapper_key=wrapper_key
    )

    reference_counts = anneal_result["best_reference_counts"]
    ood_counts = anneal_result["best_ood_counts"]
    target_delta = anneal_result["target_delta"]

    phi_ref = compute_phi(reference_counts)
    phi_ood = compute_phi(ood_counts)
    actual_delta = np.abs(phi_ref - phi_ood)
    objective_error = np.abs(actual_delta - target_delta)

    metadata = {
        "description": "OOD test subset constructed from the official SDK test split.",
        "config": asdict(cfg),
        "source_test_path": cfg.test_path,
        "source_test_summary": {
            "num_samples": int(len(labels)),
            "num_videos": int(num_videos),
            "label_count": label_count(labels),
            "objective_class_count": class_count_for_objective(labels, cfg),
        },
        "reference_subset_summary": summarize_subset(
            split_data=test_data,
            labels=labels,
            sample_indices=reference_sample_indices,
            cfg=cfg
        ),
        "ood_subset_summary": summarize_subset(
            split_data=test_data,
            labels=labels,
            sample_indices=ood_sample_indices,
            cfg=cfg
        ),
        "objective": {
            "best_value": float(anneal_result["best_value"]),
            "target_delta": float(cfg.target_delta),
            "actual_delta_mean": float(actual_delta.mean()),
            "actual_delta_std": float(actual_delta.std()),
            "actual_delta_max": float(actual_delta.max()),
            "objective_error_mean": float(objective_error.mean()),
            "objective_error_std": float(objective_error.std()),
            "objective_error_max": float(objective_error.max()),
        },
        "video_ids": {
            "ood_test": [video_ids[int(i)] for i in ood_video_indices],
            "reference_test": [video_ids[int(i)] for i in reference_video_indices],
        },
        "sample_indices": {
            "ood_test": [int(i) for i in ood_sample_indices.tolist()],
            "reference_test": [int(i) for i in reference_sample_indices.tolist()],
        },
        "top_shift_words": compute_distribution_table(
            reference_counts=reference_counts,
            ood_counts=ood_counts,
            vocab=vocab,
            top_k=30
        ),
        "annealing_history_last_20": anneal_result["history"][-20:],
        "output_files": {
            "ood_pickle": str(output_ood_path),
            "metadata_json": str(output_meta_path),
        }
    }

    save_pickle(output_ood_data, str(output_ood_path))
    save_json(metadata, str(output_meta_path))

    print("=" * 80)
    print("Final OOD subset summary")
    print(metadata["ood_subset_summary"])
    print("=" * 80)
    print("Reference subset summary")
    print(metadata["reference_subset_summary"])
    print("=" * 80)
    print("Saved files")
    print(f"OOD pickle   : {output_ood_path}")
    print(f"Metadata JSON: {output_meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    main(CONFIG)
