from .centroid_strategy import CentroidAssignmentStragety
import numpy as np

from .centroid_strategy import CentroidAssignmentStragety
import numpy as np
import faiss


class RandomAssignmentStrategy_opq(CentroidAssignmentStragety):
    """
    Random embeddings + optional OPQ + PQ
    Returns:
        item_codes: [num_items, M]
        centroids: [M, Ks, D_sub]
    """

    def __init__(self, item_code_bytes, num_items, sub_embedding_size,
                 vals_per_dim=256, use_opq=True, normalize_embeddings=False):

        self.item_code_bytes = item_code_bytes
        self.num_items = num_items
        self.sub_embedding_size = sub_embedding_size
        self.vals_per_dim = vals_per_dim
        self.use_opq = use_opq
        self.normalize_embeddings = normalize_embeddings

    def assign(self, train_users=None):
        # -----------------------------
        # Step 1: Random embeddings
        # -----------------------------
        d = self.item_code_bytes * self.sub_embedding_size

        X = np.random.randn(self.num_items+1, d).astype(np.float32)

        # -----------------------------
        # Step 2: Dimension check
        # -----------------------------
        if d % self.item_code_bytes != 0:
            raise ValueError(f"d={d} must be divisible by M={self.item_code_bytes}")

        # -----------------------------
        # Step 3: OPQ
        # -----------------------------
        if self.use_opq:
            opq = faiss.OPQMatrix(d, self.item_code_bytes)
            opq.train(X)
            X = opq.apply_py(X)

        # -----------------------------
        # Step 4: Normalize (optional)
        # -----------------------------
        if self.normalize_embeddings:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / (norms + 1e-12)

        # -----------------------------
        # Step 5: PQ
        # -----------------------------
        nbits = int(np.log2(self.vals_per_dim))
        pq = faiss.ProductQuantizer(d, self.item_code_bytes, nbits)
        pq.train(X)

        # -----------------------------
        # Step 6: Codes
        # -----------------------------
        item_codes = pq.compute_codes(X)

        # -----------------------------
        # Step 7: Centroids
        # -----------------------------
        d_sub = d // self.item_code_bytes
        centroids_flat = faiss.vector_to_array(pq.centroids)
        centroids = centroids_flat.reshape(
            self.item_code_bytes, self.vals_per_dim, d_sub
        )

        # -----------------------------
        # Step 8: Distortion
        # -----------------------------
        reconstructed = pq.decode(item_codes)
        distortion = np.mean(np.sum((X - reconstructed) ** 2, axis=1))
        print(f"[Random OPQ] Quantization distortion: {distortion:.6f}")

        return item_codes.astype(np.uint8), centroids