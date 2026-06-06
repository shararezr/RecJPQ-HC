import numpy as np
import torch
import tqdm as tqdm_module
from ir_measures import ScoredDoc, Qrel
import ir_measures


def compute_quantization_error(model, original_embeddings=None):
    """
    Compute reconstruction error of PQ codes vs centroid-based embeddings.

    If original_embeddings is provided, computes MSE between original and
    reconstructed. Otherwise computes stats about the reconstructed embeddings.

    Args:
        model: the GSASRec model (must have item_embedding with item_codes and centroids)
        original_embeddings: [num_items, D] numpy array of SVD embeddings before quantization

    Returns:
        dict with quantization metrics
    """
    layer = model.item_embedding
    if not hasattr(layer, 'item_codes') or not hasattr(layer, 'centroids'):
        return None

    num_items = layer.num_items
    M = layer.item_code_bytes
    d_sub = layer.sub_embedding_size

    codes = layer.item_codes[:num_items].cpu().numpy()        # [num_items, M]
    centroids = layer.centroids.detach().cpu().numpy()         # [M, Ks, d_sub]

    # Reconstruct embeddings from codes + centroids
    reconstructed = np.zeros((num_items, M * d_sub), dtype=np.float32)
    for m in range(M):
        reconstructed[:, m*d_sub:(m+1)*d_sub] = centroids[m, codes[:, m], :]

    result = {}

    if original_embeddings is not None:
        # MSE between original and reconstructed
        mse = np.mean((original_embeddings - reconstructed) ** 2)
        result["mse"] = float(mse)
        result["rmse"] = float(np.sqrt(mse))

    # Code utilization: how many of the 256 codes are actually used per subspace
    code_utilization = []
    for m in range(M):
        n_unique = len(np.unique(codes[:, m]))
        code_utilization.append(n_unique)
    result["mean_code_utilization"] = float(np.mean(code_utilization))
    result["code_utilization_per_subspace"] = code_utilization

    return result


def compute_effective_rank(model):
    """
    Compute effective rank of the reconstructed item embedding matrix.
    Effective rank = exp(entropy of normalized singular values).

    Args:
        model: the GSASRec model

    Returns:
        dict with effective rank metrics
    """
    layer = model.item_embedding
    if not hasattr(layer, 'item_codes') or not hasattr(layer, 'centroids'):
        return None

    num_items = layer.num_items
    M = layer.item_code_bytes
    d_sub = layer.sub_embedding_size

    codes = layer.item_codes[:num_items].cpu().numpy()
    centroids = layer.centroids.detach().cpu().numpy()

    # Reconstruct embeddings
    reconstructed = np.zeros((num_items, M * d_sub), dtype=np.float32)
    for m in range(M):
        reconstructed[:, m*d_sub:(m+1)*d_sub] = centroids[m, codes[:, m], :]

    # SVD
    _, s, _ = np.linalg.svd(reconstructed, full_matrices=False)

    # Normalized singular values
    s_norm = s / s.sum()
    s_norm = s_norm[s_norm > 0]

    # Entropy -> effective rank
    entropy = -np.sum(s_norm * np.log(s_norm))
    effective_rank = np.exp(entropy)

    return {
        "effective_rank": float(effective_rank),
        "embedding_dim": int(M * d_sub),
        "top5_singular_values": s[:5].tolist(),
    }


def compute_item_popularity(dataset_name, config):
    """
    Count item frequency in training data. Returns normalized popularity array.
    """
    from dataset_utils import get_train_dataloader, get_num_items
    num_items = get_num_items(dataset_name)
    train_loader = get_train_dataloader(
        dataset_name,
        batch_size=config.train_batch_size,
        max_length=config.sequence_length,
        train_neg_per_positive=config.negs_per_pos
    )
    item_freq = np.zeros(num_items)
    for positives, _ in train_loader:
        for seq in positives.numpy():
            for i in seq:
                if 0 <= i < num_items:
                    item_freq[i] += 1
    popularity = item_freq / (item_freq.max() + 1e-9)
    return popularity


def get_cold_start_items(popularity, percentile=10):
    """Return item indices in the bottom percentile of popularity."""
    threshold = np.percentile(popularity, percentile)
    return set(np.where(popularity <= threshold)[0])


def evaluate_coldstart(model, data_loader, metrics, limit, filter_rated,
                       cold_items, device):
    """
    Evaluate model separately on cold vs warm test targets.
    Returns dict with cold_* and warm_* metrics.
    """
    model.eval()

    cold_scored = []
    cold_qrels = []
    warm_scored = []
    warm_qrels = []
    cold_count = 0
    warm_count = 0

    with torch.no_grad():
        for batch_idx, (data, rated, target) in tqdm_module.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Cold-start eval"
        ):
            data, target = data.to(device), target.to(device)
            if filter_rated:
                items, scores = model.get_predictions(data, limit, rated)
            else:
                items, scores = model.get_predictions(data, limit)

            for rec_items, rec_scores, t in zip(items, scores, target):
                t_item = t.item()
                is_cold = t_item in cold_items

                uid = str(cold_count if is_cold else warm_count)
                for item, score in zip(rec_items, rec_scores):
                    doc = ScoredDoc(uid, str(item.item()), score.item())
                    if is_cold:
                        cold_scored.append(doc)
                    else:
                        warm_scored.append(doc)

                qrel = Qrel(uid, str(t_item), 1)
                if is_cold:
                    cold_qrels.append(qrel)
                    cold_count += 1
                else:
                    warm_qrels.append(qrel)
                    warm_count += 1

    result = {"cold_count": cold_count, "warm_count": warm_count}

    if cold_qrels:
        cold_result = ir_measures.calc_aggregate(metrics, cold_qrels, cold_scored)
        for k, v in cold_result.items():
            result[f"cold_{k}"] = float(v)

    if warm_qrels:
        warm_result = ir_measures.calc_aggregate(metrics, warm_qrels, warm_scored)
        for k, v in warm_result.items():
            result[f"warm_{k}"] = float(v)

    return result
