
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import KBinsDiscretizer
from .centroid_strategy import CentroidAssignmentStragety
from sklearn.cluster import KMeans
import numpy as np
import torch
import scipy.sparse as sp



import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import faiss
from .centroid_strategy import CentroidAssignmentStragety

faiss.omp_set_num_threads(1)
# This is the best up to now and then I want to the inequality sign to the equality sign.
# I want to change SVD_embedding from num_items to num_items+1
import torch, numpy as np, random, faiss
from sklearn.preprocessing import normalize
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
from sklearn.preprocessing import normalize



class SVDAssignmentStrategy_opq(CentroidAssignmentStragety):
    """
    SVD + optional OPQ rotation + PQ per subspace using FAISS.
    Returns:
        item_codes: [num_items + 1, M] uint8 (extra row for padding)
        centroids: [M, Ks, D_sub]
    """

    def __init__(self, item_code_bytes, num_items, sub_embedding_size,
                 vals_per_dim=256, use_opq=False, normalize_embeddings=False):
        self.item_code_bytes = item_code_bytes  # M
        self.num_items = num_items
        self.sub_embedding_size = sub_embedding_size
        self.vals_per_dim = vals_per_dim       # Ks
        self.use_opq = use_opq
        self.normalize_embeddings = normalize_embeddings

    def assign(self, train_users=None, svd_embeddings=None):
        # -----------------------------
        # Step 1: Build item embeddings
        # -----------------------------
        if svd_embeddings is None:
            # Build user-item sparse matrix
            rows, cols, vals = [], [], []
            for u, seq in enumerate(train_users):
                for item in seq:
                    if 0 <= item < self.num_items:
                        rows.append(u)
                        cols.append(item)
                        vals.append(1)


            matr = csr_matrix((vals, (rows, cols)), shape=(len(train_users), self.num_items))
            '''
            user_deg = np.array(matr.sum(axis=1)).flatten()
            item_deg = np.array(matr.sum(axis=0)).flatten()
            user_deg[user_deg == 0] = 1
            item_deg[item_deg == 0] = 1
            matr = matr.multiply(1.0 / np.sqrt(user_deg)[:, None])
            matr = matr.multiply(1.0 / np.sqrt(item_deg)[None, :])
            '''


            # --- SVD ---
            d = self.item_code_bytes * self.sub_embedding_size
            svd = TruncatedSVD(n_components=d, random_state=42)
            svd_embeddings = svd.fit_transform(matr.T).astype(np.float32)  # [num_items, d]

        # -----------------------------
        # Step 2: Check dimensions
        # -----------------------------
        num_items_emb, d = svd_embeddings.shape
        #if num_items_emb != self.num_items:
        #    raise ValueError(f"SVD embeddings rows {num_items_emb} != num_items {self.num_items}")
        if d % self.item_code_bytes != 0:
            raise ValueError(f"SVD dimension {d} must be divisible by M={self.item_code_bytes}")

        # Optional global L2 normalization (cosine similarity)


        # -----------------------------
        # Step 3: Apply OPQ rotation
        # -----------------------------
        opq = None
        if self.use_opq:
            opq = faiss.OPQMatrix(d, self.item_code_bytes)
            opq.train(svd_embeddings)
            svd_embeddings = opq.apply_py(svd_embeddings)



        if self.normalize_embeddings:
            norms = np.linalg.norm(svd_embeddings, axis=1, keepdims=True)
            svd_embeddings = svd_embeddings / (norms + 1e-12)

        # -----------------------------
        # Step 4: Train Product Quantizer (PQ)
        # -----------------------------
        nbits = int(np.log2(self.vals_per_dim))
        pq = faiss.ProductQuantizer(d, self.item_code_bytes, nbits)
        pq.train(svd_embeddings)

        # -----------------------------
        # Step 5: Encode items into PQ codes
        # -----------------------------
        item_codes = pq.compute_codes(svd_embeddings)  # [num_items, M], uint8
        # Add extra padding row
        item_codes = np.vstack([item_codes, np.zeros((1, self.item_code_bytes), dtype=np.uint8)])

        # -----------------------------
        # Step 6: Extract centroids per subspace
        # -----------------------------
        d_sub = d // self.item_code_bytes
        centroids_flat = faiss.vector_to_array(pq.centroids)
        centroids = centroids_flat.reshape(self.item_code_bytes, self.vals_per_dim, d_sub)

        # -----------------------------
        # Step 7: Return
        # -----------------------------
        return item_codes,centroids


