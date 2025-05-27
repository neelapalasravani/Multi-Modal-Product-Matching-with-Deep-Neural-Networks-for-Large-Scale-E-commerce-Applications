import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FeatureFusion:
    def __init__(self, apply_scaling=False, apply_pca=False, n_components=50):
        """
        apply_scaling: bool - Whether to standardize features before concatenation
        apply_pca: bool - Whether to apply PCA dimensionality reduction after concatenation
        n_components: int - Number of PCA components if apply_pca is True
        """
        self.apply_scaling = apply_scaling
        self.apply_pca = apply_pca
        self.n_components = n_components
        self.scaler = StandardScaler() if apply_scaling else None
        self.pca = PCA(n_components=n_components) if apply_pca else None

    def combine_features(self, *feature_arrays):
        """
        Horizontally concatenate multiple numpy arrays of features
        """
        combined = np.hstack(feature_arrays)
        if self.apply_scaling:
            combined = self.scaler.fit_transform(combined)
        if self.apply_pca:
            combined = self.pca.fit_transform(combined)
        return combined

    def transform_features(self, *feature_arrays):
        """
        Use after fit, transform new data (e.g., test set) using fitted scaler and PCA
        """
        combined = np.hstack(feature_arrays)
        if self.apply_scaling:
            combined = self.scaler.transform(combined)
        if self.apply_pca:
            combined = self.pca.transform(combined)
        return combined
