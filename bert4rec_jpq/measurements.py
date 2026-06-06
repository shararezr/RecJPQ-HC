import numpy as np
import torch
import tqdm as tqdm_module
from torch.utils.data import DataLoader
from ir_measures import ScoredDoc, Qrel
import ir_measures

'''
def compute_quantization_error(model, original_embeddings=None):
    """
    Compute reconstruction error of PQ codes vs centroid-based embeddings.
    """
    layer = model.item_embedding
    if not hasattr(layer, 'item_codes') or not hasattr(layer, 'centroids'):
        return None

    num_items = layer.num_items
    M = layer.item_code_bytes
    d_sub = layer.sub_embedding_size

    codes = layer.item_codes[:num_items].cpu().numpy()
    centroids = layer.centroids.detach().cpu().numpy()

    reconstructed = np.zeros((num_items, M * d_sub), dtype=np.float32)
    for m in range(M):
        reconstructed[:, m*d_sub:(m+1)*d_sub] = centroids[m, codes[:, m], :]

    result = {}

    if original_embeddings is not None:
        mse = np.mean((original_embeddings - reconstructed) ** 2)
        result["mse"] = float(mse)
        result["rmse"] = float(np.sqrt(mse))

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
    """
    layer = model.item_embedding
    if not hasattr(layer, 'item_codes') or not hasattr(layer, 'centroids'):
        return None

    num_items = layer.num_items
    M = layer.item_code_bytes
    d_sub = layer.sub_embedding_size

    codes = layer.item_codes[:num_items].cpu().numpy()
    centroids = layer.centroids.detach().cpu().numpy()

    reconstructed = np.zeros((num_items, M * d_sub), dtype=np.float32)
    for m in range(M):
        reconstructed[:, m*d_sub:(m+1)*d_sub] = centroids[m, codes[:, m], :]

    _, s, _ = np.linalg.svd(reconstructed, full_matrices=False)

    s_norm = s / s.sum()
    s_norm = s_norm[s_norm > 0]

    entropy = -np.sum(s_norm * np.log(s_norm))
    effective_rank = np.exp(entropy)

    return {
        "effective_rank": float(effective_rank),
        "embedding_dim": int(M * d_sub),
        "top5_singular_values": s[:5].tolist(),
    }


def compute_item_popularity(recommender):
    """
    Count item frequency from user_actions (bert4rec data format).
    Returns normalized popularity array.
    """
    num_items = recommender.items.size()
    item_freq = np.zeros(num_items)
    for uid, actions in recommender.user_actions.items():
        for _, item_id in actions:
            if 0 <= item_id < num_items:
                item_freq[item_id] += 1
    popularity = item_freq / (item_freq.max() + 1e-9)
    return popularity


def get_cold_start_items(popularity, percentile=10):
    """Return item indices in the bottom percentile of popularity."""
    threshold = np.percentile(popularity, percentile)
    return set(np.where(popularity <= threshold)[0])


def evaluate_coldstart(recommender, cold_items, limit, mode='val'):
    """
    Evaluate model separately on cold vs warm test targets.
    Uses the recommender's existing collator and recommend_impl.
    """
    recommender.model.eval()

    if mode == 'val':
        user_ids = list(recommender.val_users)
    else:
        user_ids = list(recommender.val_users)  # adjust if test users differ

    cold_hits = []
    warm_hits = []

    with torch.no_grad():
        for batch_uids in DataLoader(user_ids, batch_size=recommender.val_batch_size, shuffle=False):
            result = recommender.recommend_impl(batch_uids, limit, mode)
            items = result['items'].cpu()

            if mode == 'val':
                labels = result['labels'].cpu()
                for i in range(len(labels)):
                    target = labels[i].item()
                    predicted = items[i].tolist()
                    hit = 1.0 if target in predicted else 0.0

                    if target in cold_items:
                        cold_hits.append(hit)
                    else:
                        warm_hits.append(hit)

    results = {}
    if cold_hits:
        results["cold_recall"] = float(np.mean(cold_hits))
        results["cold_count"] = len(cold_hits)
    if warm_hits:
        results["warm_recall"] = float(np.mean(warm_hits))
        results["warm_count"] = len(warm_hits)

    return results

'''
import numpy as np
import torch
from torch.utils.data import DataLoader


# ============================================================
# QUANTIZATION ERROR
# ============================================================

def compute_quantization_error(model, original_embeddings=None):
    """
    Compute reconstruction error of PQ codes vs centroid-based embeddings.

    Returns:
        - mse (per-dimension error)
        - rmse
        - distortion (PQ-style per-vector L2 error)
        - code utilization stats
    """
    layer = model.item_embedding
    if not hasattr(layer, 'item_codes') or not hasattr(layer, 'centroids'):
        return None

    num_items = layer.num_items
    M = layer.item_code_bytes
    d_sub = layer.sub_embedding_size

    codes = layer.item_codes[:num_items].cpu().numpy()
    centroids = layer.centroids.detach().cpu().numpy()

    # -------------------------
    # reconstruct embeddings
    # -------------------------
    reconstructed = np.zeros((num_items, M * d_sub), dtype=np.float32)

    for m in range(M):
        reconstructed[:, m*d_sub:(m+1)*d_sub] = centroids[m, codes[:, m], :]

    result = {}

    if original_embeddings is not None:
        diff = original_embeddings - reconstructed

        # Mean Squared Error (per dimension)
        mse = np.mean(diff ** 2)

        # Root MSE
        rmse = np.sqrt(mse)

        # PQ-style distortion (per vector L2 squared)
        distortion = np.mean(np.sum(diff ** 2, axis=1))

        result["mse"] = float(mse)
        result["rmse"] = float(rmse)
        result["distortion"] = float(distortion)

    # -------------------------
    # code utilization
    # -------------------------
    code_utilization = []
    for m in range(M):
        n_unique = len(np.unique(codes[:, m]))
        code_utilization.append(n_unique)

    result["mean_code_utilization"] = float(np.mean(code_utilization))
    result["code_utilization_per_subspace"] = code_utilization

    return result


# ============================================================
# EFFECTIVE RANK (GEOMETRY-CORRECT)
# ============================================================
'''
def compute_effective_rank(model):
    """
    Compute effective rank using entropy of singular values.

    Returns:
        - effective_rank
        - normalized_effective_rank
        - spectral_dominance
        - information_abundance
        - embedding_dim
        - top singular values
    """
    layer = model.item_embedding
    if not hasattr(layer, 'item_codes') or not hasattr(layer, 'centroids'):
        return None

    num_items = layer.num_items
    M = layer.item_code_bytes
    d_sub = layer.sub_embedding_size

    codes = layer.item_codes[:num_items].cpu().numpy()
    centroids = layer.centroids.detach().cpu().numpy()

    # -------------------------
    # reconstruct embeddings
    # -------------------------
    reconstructed = np.zeros((num_items, M * d_sub), dtype=np.float32)

    for m in range(M):
        reconstructed[:, m*d_sub:(m+1)*d_sub] = centroids[m, codes[:, m], :]

    # -------------------------
    # CENTERING (critical!)
    # -------------------------
    reconstructed = reconstructed - reconstructed.mean(axis=0, keepdims=True)

    # -------------------------
    # SVD
    # -------------------------
    _, s, _ = np.linalg.svd(reconstructed, full_matrices=False)

    s_sum = s.sum() + 1e-9
    p = s / s_sum

    # -------------------------
    # effective rank
    # -------------------------
    entropy = -np.sum(p * np.log(p + 1e-9))
    effective_rank = np.exp(entropy)

    # -------------------------
    # additional geometry metrics
    # -------------------------
    spectral_dominance = s[0] / s_sum
    information_abundance = s_sum / (s[0] + 1e-9)

    embedding_dim = int(M * d_sub)
    normalized_rank = effective_rank / embedding_dim

    return {
        "effective_rank": float(effective_rank),
        "normalized_effective_rank": float(normalized_rank),
        "spectral_dominance": float(spectral_dominance),
        "information_abundance": float(information_abundance),
        "embedding_dim": embedding_dim,
        "top5_singular_values": s[:5].tolist(),
    }

'''


def compute_effective_rank1(model):
    """
    Compute effective rank of reconstructed item embeddings
    using centered embeddings (same as file 2 approach).
    """
    layer = model.item_embedding
    if not hasattr(layer, 'item_codes') or not hasattr(layer, 'centroids'):
        return None

    num_items = layer.num_items
    M = layer.item_code_bytes
    d_sub = layer.sub_embedding_size

    codes = layer.item_codes[:num_items].cpu().numpy()
    centroids = layer.centroids.detach().cpu().numpy()

    reconstructed = np.zeros((num_items, M * d_sub), dtype=np.float32)

    for m in range(M):
        reconstructed[:, m * d_sub:(m + 1) * d_sub] = centroids[m, codes[:, m], :]

    # =========================
    # FILE 2 STYLE CHANGE: CENTERING
    # =========================
    mean = reconstructed.mean(axis=0, keepdims=True)
    centered = reconstructed - mean

    # SVD on centered embeddings
    _, s, _ = np.linalg.svd(centered, full_matrices=False)

    # normalize singular values (stable version like file 2)
    s = s / (s.sum() + 1e-9)
    s = s[s > 0]

    entropy = -np.sum(s * np.log(s + 1e-9))
    effective_rank = np.exp(entropy)

    return {
        "effective_rank": float(effective_rank),
        "embedding_dim": int(M * d_sub),
        "top5_singular_values": s[:5].tolist(),
    }




def compute_effective_rank(model):
    """
    Now returns Information Abundance (IA) instead of effective rank,
    using the same centered embedding representation.
    """
    layer = model.item_embedding
    if not hasattr(layer, 'item_codes') or not hasattr(layer, 'centroids'):
        return None

    num_items = layer.num_items
    M = layer.item_code_bytes
    d_sub = layer.sub_embedding_size

    codes = layer.item_codes[:num_items].cpu().numpy()
    centroids = layer.centroids.detach().cpu().numpy()

    reconstructed = np.zeros((num_items, M * d_sub), dtype=np.float32)

    for m in range(M):
        reconstructed[:, m * d_sub:(m + 1) * d_sub] = centroids[m, codes[:, m], :]

    # =========================
    # CENTERING (same as before)
    # =========================
    mean = reconstructed.mean(axis=0, keepdims=True)
    centered = reconstructed - mean

    # =========================
    # SVD
    # =========================
    _, s, _ = np.linalg.svd(centered, full_matrices=False)

    # avoid division issues
    s_sum = s.sum() + 1e-9

    # =========================
    # INFORMATION ABUNDANCE (IA)
    # =========================
    ia = s_sum / (s[0] + 1e-9)

    # optional stability view (same info as before but renamed meaningfully)
    spectral_dominance = s[0] / s_sum

    return {
        "effective_rank": float(ia),
        "embedding_dim": int(M * d_sub),
        "top5_singular_values": (s / s_sum)[:5].tolist()
    }

# ============================================================
# POPULARITY
# ============================================================

def compute_item_popularity(recommender):
    """
    Count item frequency from user_actions (bert4rec data format).
    Returns normalized popularity array.
    """
    num_items = recommender.items.size()
    item_freq = np.zeros(num_items)

    for _, actions in recommender.user_actions.items():
        for _, item_id in actions:
            if 0 <= item_id < num_items:
                item_freq[item_id] += 1

    popularity = item_freq / (item_freq.max() + 1e-9)
    return popularity


# ============================================================
# COLD START
# ============================================================

def get_cold_start_items(popularity, percentile=10):
    """Return item indices in the bottom percentile of popularity."""
    threshold = np.percentile(popularity, percentile)
    return set(np.where(popularity <= threshold)[0])


def evaluate_coldstart(recommender, cold_items, limit, mode='val'):
    """
    Evaluate model separately on cold vs warm items.
    """
    recommender.model.eval()

    if mode == 'val':
        user_ids = list(recommender.val_users)
    else:
        user_ids = list(recommender.val_users)

    cold_hits = []
    warm_hits = []

    with torch.no_grad():
        for batch_uids in DataLoader(
            user_ids,
            batch_size=recommender.val_batch_size,
            shuffle=False
        ):
            result = recommender.recommend_impl(batch_uids, limit, mode)
            items = result['items'].cpu()

            if mode == 'val':
                labels = result['labels'].cpu()

                for i in range(len(labels)):
                    target = labels[i].item()
                    predicted = items[i].tolist()

                    hit = 1.0 if target in predicted else 0.0

                    if target in cold_items:
                        cold_hits.append(hit)
                    else:
                        warm_hits.append(hit)

    results = {}

    if cold_hits:
        results["cold_recall"] = float(np.mean(cold_hits))
        results["cold_count"] = len(cold_hits)

    if warm_hits:
        results["warm_recall"] = float(np.mean(warm_hits))
        results["warm_count"] = len(warm_hits)

    return results
