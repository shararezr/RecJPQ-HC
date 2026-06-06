from lightfm import LightFM
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import KBinsDiscretizer
from .centroid_strategy import CentroidAssignmentStragety
import numpy as np

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import faiss
from sklearn.preprocessing import normalize



class BPRAssignmentStrategy(CentroidAssignmentStragety):
    def assign(self, train_users):
        rows = []
        cols = []
        vals = []
        for i in range(len(train_users)):
            for j in range(len(train_users[i])):
                rows.append(i)
                cols.append(train_users[i][j])
                vals.append(1)
        matr = csr_matrix((vals, [rows, cols]), shape=(len(train_users), self.num_items+2)) # +2 for padding and mask
        print("fitting MF-BPR for initial centroids assignments")

        model = LightFM(no_components=self.item_code_bytes, loss='bpr')
        model.fit(matr, epochs=20, verbose=True, num_threads=20)
        item_embeddings = model.get_item_representations()[1].T

        assignments = []
        print("done")
        for i in range(self.item_code_bytes):
            discretizer = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile')
            ith_component = item_embeddings[i:i+1][0]
            ith_component = (ith_component - np.min(ith_component))/(np.max(ith_component) - np.min(ith_component) + 1e-10)
            noise = np.random.normal(0, 1e-5, self.num_items + 2)
            ith_component += noise # make sure that every item has unique value
            ith_component = np.expand_dims(ith_component, 1)
            component_assignments = discretizer.fit_transform(ith_component).astype('uint8')[:,0]
            assignments.append(component_assignments)
        return np.transpose(np.array(assignments))










class BPRAssignmentStrategy_(CentroidAssignmentStragety):

    def __init__(self, item_code_bytes, num_items, sub_embedding_size,
                 vals_per_dim=256, use_opq=True, normalize_embeddings=False):
        self.item_code_bytes = item_code_bytes  # M
        self.num_items = num_items
        self.sub_embedding_size = sub_embedding_size
        self.vals_per_dim = vals_per_dim       # Ks
        self.use_opq = use_opq
        self.normalize_embeddings = normalize_embeddings


    def assign(self, train_users):
        rows = []
        cols = []
        vals = []
        for i in range(len(train_users)):
            for j in range(len(train_users[i])):
                rows.append(i)
                cols.append(train_users[i][j])
                vals.append(1)
        matr = csr_matrix((vals, [rows, cols]), shape=(len(train_users), self.num_items+1)) # +2 for padding and mask

        print("fitting MF-BPR for initial centroids assignments")
        d = self.item_code_bytes * self.sub_embedding_size
        model = LightFM(no_components=d, loss='bpr')
        model.fit(matr, epochs=20, verbose=True, num_threads=20)
        item_embeddings = model.get_item_representations()[1]
                # -----------------------------
        # Step 2: Check dimensions
        # -----------------------------
        num_items_emb, d = item_embeddings.shape
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
            opq.train(item_embeddings)
            item_embeddings = opq.apply_py(item_embeddings)



        if self.normalize_embeddings:
            norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
            item_embeddings = item_embeddings / (norms + 1e-12)

        # -----------------------------
        # Step 4: Train Product Quantizer (PQ)
        # -----------------------------
        nbits = int(np.log2(self.vals_per_dim))
        pq = faiss.ProductQuantizer(d, self.item_code_bytes, nbits)
        pq.train(item_embeddings)

        # -----------------------------
        # Step 5: Encode items into PQ codes
        # -----------------------------
        item_codes = pq.compute_codes(item_embeddings)  # [num_items, M], uint8
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
        return item_codes, centroids

