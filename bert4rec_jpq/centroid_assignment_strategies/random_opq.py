from .centroid_strategy import CentroidAssignmentStragety
import numpy as np
import faiss



class RandomAssignmentStrategy_opq(CentroidAssignmentStragety):
    """
    Random embeddings + optional OPQ + PQ per subspace using FAISS.

    Returns:
        item_codes: [num_items + 1, M] uint8
        centroids: [M, Ks, D_sub]
    """

    def __init__(
        self,
        item_code_bytes,
        num_items,
        sub_embedding_size,
        vals_per_dim=256,
        use_opq=True,
        normalize_embeddings=False,
        seed=42
    ):

        self.item_code_bytes = item_code_bytes   # M
        self.num_items = num_items
        self.sub_embedding_size = sub_embedding_size
        self.vals_per_dim = vals_per_dim         # Ks
        self.use_opq = use_opq
        self.normalize_embeddings = normalize_embeddings
        self.seed = seed

    def assign(self, train_users=None, random_embeddings=None):

        # -------------------------------------------------
        # Step 1: Generate random embeddings
        # -------------------------------------------------

        if random_embeddings is None:

            np.random.seed(self.seed)

            d = self.item_code_bytes * self.sub_embedding_size

            random_embeddings = np.random.randn(
                self.num_items,
                d
            ).astype(np.float32)

        # -------------------------------------------------
        # Step 2: Check dimensions
        # -------------------------------------------------

        num_items_emb, d = random_embeddings.shape

        if d % self.item_code_bytes != 0:
            raise ValueError(
                f"Embedding dimension {d} "
                f"must be divisible by M={self.item_code_bytes}"
            )

        # -------------------------------------------------
        # Step 3: Apply OPQ rotation
        # -------------------------------------------------

        opq = None

        if self.use_opq:
            print("Training OPQ...")

            opq = faiss.OPQMatrix(d, self.item_code_bytes)
            opq.train(random_embeddings)

            random_embeddings = opq.apply_py(random_embeddings)

        # -------------------------------------------------
        # Step 4: Optional normalization
        # -------------------------------------------------

        if self.normalize_embeddings:
            norms = np.linalg.norm(
                random_embeddings,
                axis=1,
                keepdims=True
            )

            random_embeddings = (
                random_embeddings / (norms + 1e-12)
            )

        # -------------------------------------------------
        # Step 5: Train Product Quantizer (PQ)
        # -------------------------------------------------

        nbits = int(np.log2(self.vals_per_dim))

        pq = faiss.ProductQuantizer(
            d,
            self.item_code_bytes,
            nbits
        )

        print("Training PQ...")
        pq.train(random_embeddings)

        # -------------------------------------------------
        # Step 6: Encode embeddings into PQ codes
        # -------------------------------------------------

        item_codes = pq.compute_codes(random_embeddings)


        # -------------------------------------------------
        # Step 7: Extract centroids
        # -------------------------------------------------

        d_sub = d // self.item_code_bytes

        centroids_flat = faiss.vector_to_array(pq.centroids)

        centroids = centroids_flat.reshape(
            self.item_code_bytes,
            self.vals_per_dim,
            d_sub
        )

        # -------------------------------------------------
        # Step 8: Quantization distortion
        # -------------------------------------------------

        reconstructed_embeddings = pq.decode(
            item_codes[:-1]   # remove padding row
        )

        distortion = np.mean(
            np.sum(
                (random_embeddings - reconstructed_embeddings) ** 2,
                axis=1
            )
        )

        print(f"[Random OPQ] Quantization distortion: {distortion:.6f}")

        # -------------------------------------------------
        # Step 9: Return
        # -------------------------------------------------

        return item_codes, centroids