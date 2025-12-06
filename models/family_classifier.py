import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from config.config import FAMILY_THRESHOLD_PERCENTILE, FAMILY_THRESHOLD_MULTIPLIER

class FamilyClassifier:
    def __init__(self):
        self.centroids = {}
        self.thresholds = {}
        self.family_names = {}
        self.scaler = None

    def fit(self, features, labels, family_names_map):
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue
            mask = labels == label
            cluster_features = features_scaled[mask]
            centroid = np.mean(cluster_features, axis=0)
            self.centroids[int(label)] = centroid
            dists = np.linalg.norm(cluster_features - centroid, axis=1)
            limit_dist = np.percentile(dists, FAMILY_THRESHOLD_PERCENTILE) if len(dists) > 0 else 0
            self.thresholds[int(label)] = limit_dist * FAMILY_THRESHOLD_MULTIPLIER if limit_dist > 0 else 1.0
            self.family_names[int(label)] = family_names_map.get(int(label), f"Family_{label}")

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'centroids': self.centroids,
                'thresholds': self.thresholds,
                'family_names': self.family_names,
                'scaler': self.scaler
            }, f)

    def load(self, path):
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.centroids = data['centroids']
                self.thresholds = data['thresholds']
                self.family_names = data['family_names']
                self.scaler = data.get('scaler')
            return True
        except Exception:
            return False

    def predict(self, feature_vector):
        if not self.centroids:
            return None, "Model_Not_Loaded", True
        if self.scaler:
            feature_vector = self.scaler.transform([feature_vector])[0]
        min_dist = float('inf')
        best_label = None
        for label, centroid in self.centroids.items():
            dist = np.linalg.norm(feature_vector - centroid)
            if dist < min_dist:
                min_dist = dist
                best_label = label
        if best_label is not None:
            threshold = self.thresholds[best_label]
            if min_dist <= threshold:
                return best_label, self.family_names[best_label], False
        return None, "New_Unknown_Family", True