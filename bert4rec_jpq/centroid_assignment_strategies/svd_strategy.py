
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import KBinsDiscretizer
from .centroid_strategy import CentroidAssignmentStragety
import numpy as np

class SVDAssignmentStrategy(CentroidAssignmentStragety):
    def assign(self, train_users):
        rows = []
        cols = []
        vals = []
        for i in range(len(train_users)):
            for j in range(len(train_users[i])):
                rows.append(i)
                cols.append(train_users[i][j])
                vals.append(1)
        matr = csr_matrix((vals, [rows, cols]), shape=(len(train_users), self.num_items+2)) # +2 for padding and mask# +2 for padding and mask
        print("fitting svd for initial centroids assignments")
        svd = TruncatedSVD(n_components=self.item_code_bytes)
        svd.fit(matr)
        item_embeddings = svd.components_
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


'''



from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import KBinsDiscretizer
from .centroid_strategy import CentroidAssignmentStragety
import numpy as np


class SVDAssignmentStrategy(CentroidAssignmentStragety):
    def assign(self, train_users):
        rows = []
        cols = []
        vals = []

        # Build user-item sparse matrix
        for u, seq in enumerate(train_users):
            for item in seq:
                if 0 <= item < self.num_items:
                    rows.append(u)
                    cols.append(item)
                    vals.append(1)
     


        matr = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(train_users), self.num_items+2)  # +2 for padding/mask, , random_state=42
        )

        print("fitting svd for initial centroids assignments")
        svd = TruncatedSVD(n_components=self.item_code_bytes)
        svd.fit(matr)
        item_embeddings = svd.components_
        print("done")

        assignments = []
        for i in range(self.item_code_bytes):
            discretizer = KBinsDiscretizer(
                n_bins=256,
                encode='ordinal',
                strategy='quantile'
            )

            ith_component = item_embeddings[i]
            ith_component = (
                ith_component - ith_component.min()
            ) / (ith_component.max() - ith_component.min() + 1e-10)

            # small noise to avoid identical bins
            ith_component += np.random.normal(0, 1e-5, self.num_items + 2)

            ith_component = ith_component[:, None]
            component_assignments = discretizer.fit_transform(
                ith_component
            ).astype("uint8")[:, 0]

            assignments.append(component_assignments)

        return np.stack(assignments, axis=1), None

