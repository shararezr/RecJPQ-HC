

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
    SVD + optional OPQ rotation + PQ per subspace.
    Returns:
        item_codes: [num_items + 1, M] uint8 (extra row for padding)
        centroids: [M, Ks, D_sub]
    """
    def __init__(self, item_code_bytes, num_items, sub_embedding_size, vals_per_dim=256, use_opq=True, normalize=False):
        super().__init__(item_code_bytes, num_items, sub_embedding_size)
        self.use_opq = use_opq
        self.normalize = normalize
        self.vals_per_dim = vals_per_dim

    def assign(self, train_users=None, svd_embeddings=None):
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
            #matr = normalize(matr, axis=1)
            #matr = normalize(matr, axis=0)
           
            # Symmetric normalization
            
            '''
            user_deg = np.array(matr.sum(axis=1)).flatten()
            item_deg = np.array(matr.sum(axis=0)).flatten()
            user_deg[user_deg == 0] = 1
            item_deg[item_deg == 0] = 1
            matr = matr.multiply(1.0 / np.sqrt(user_deg)[:, None])
            matr = matr.multiply(1.0 / np.sqrt(item_deg)[None, :])
            '''
           
            # --- SVD ---
            svd = TruncatedSVD(n_components=self.item_code_bytes * self.sub_embedding_size, random_state=42)
            svd_embeddings = svd.fit_transform(matr.T)  # [num_items, d]

        # Check dimensions
        num_items_emb, d = svd_embeddings.shape
        # if num_items_emb != self.num_items:
        #     raise ValueError(f"SVD embeddings rows {num_items_emb} != num_items {self.num_items}")
        # if d % self.item_code_bytes != 0:
        #     raise ValueError(f"SVD dimension {d} must be divisible by M={self.item_code_bytes}")



        # --- OPQ rotation ---
        if self.use_opq:
            opq = faiss.OPQMatrix(d, self.item_code_bytes)
            opq.train(svd_embeddings.astype(np.float32))
            svd_embeddings = opq.apply_py(svd_embeddings.astype(np.float32))

        if self.normalize:
            norms = np.linalg.norm(svd_embeddings, axis=1, keepdims=True)
            svd_embeddings = svd_embeddings/ (norms + 1e-12)


 

        d_sub = d // self.item_code_bytes
        M = self.item_code_bytes
        Ks = self.vals_per_dim

        # Prepare outputs
        item_codes = np.zeros((self.num_items+1, M), dtype=np.uint8)  # last row reserved for padding
        centroids_list = []

        # Split embeddings into subspaces
        for m in range(M):
            subspace = svd_embeddings[:, m*d_sub:(m+1)*d_sub]  # [num_items, d_sub]
            subspace = (subspace - subspace.min()) / (subspace.max() - subspace.min() + 1e-10)
            #subspace = normalize(subspace)
            #subspace += np.random.normal(0, 1e-6, size=subspace.shape)

            kmeans = KMeans(n_clusters=Ks, random_state=42)
            kmeans.fit(subspace)
            codes_m = kmeans.predict(subspace).astype(np.uint8)

            # Assign codes only for real items
            item_codes[:self.num_items, m] = codes_m
            centroids_list.append(kmeans.cluster_centers_.astype(np.float32))
            

        centroids = np.stack(centroids_list, axis=0)  # [M, Ks, d_sub]

        return item_codes, centroids
