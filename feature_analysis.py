import numpy as np
from enums import FeatureType
from normal import Normal
from typing import List, Dict, Optional

class FeatureAnalysis:
    def __init__(self, points: np.ndarray, half_size: int = 5):
        """
        Initialize the FeatureAnalysis class.
        :param points: Input point set as a numpy array with shape (n, 2).
        :param half_size: Half-size of the sliding window for local analysis.
        """
        if half_size <= 0:
            raise ValueError("Invalid window size")

        self.points = points
        self.half_size = half_size
        self.num_points = len(points)
        self.normals: Optional[List[Normal]] = None
        self.features: Dict[FeatureType, np.ndarray] = {}

    def compute_feature(self, feature_type: FeatureType) -> None:
        """
        Compute specific geometric features based on the provided feature type.
        :param feature_type: The type of feature to compute (Variation, Angle, etc.).
        """
        if feature_type == FeatureType.VARIATION:
            self.compute_variations()
        elif feature_type == FeatureType.DIRECT_ANGLE:
            self.compute_direct_angles()
        elif feature_type == FeatureType.DIRECT_ANGLE_DIFF:
            self.compute_direct_angle_diffs()
        elif feature_type == FeatureType.ANGLE:
            self.compute_angles()
        elif feature_type == FeatureType.ANGLE_DIFF:
            self.compute_angle_diffs()
        else:
            raise ValueError("Invalid feature type")

    def clear_cache(self, feature_type: Optional[FeatureType] = None) -> None:
        """
        Clear the cached feature calculations.
        :param feature_type: Specific feature type to clear. If None, clears all cached features.
        """
        if feature_type is None:
            self.features.clear()
        elif feature_type in self.features:
            del self.features[feature_type]

    def compute_normals(self) -> List[Normal]:
        """
        Compute normals and curve variations using PCA within a local neighborhood defined by half_size.
        :return: A list of Normal objects containing vector components and curve variation values.
        """
        if self.normals is not None:
            return self.normals

        self.normals = [Normal() for _ in range(self.num_points)]
        for i in range(self.half_size, self.num_points - self.half_size):
            start_idx = max(0, i - self.half_size)
            end_idx = min(self.num_points, i + self.half_size + 1)
            window_points = self.points[start_idx:end_idx]
            centroid = np.mean(window_points, axis=0)
            centered_points = window_points - centroid
            covariance_matrix = np.dot(centered_points.T, centered_points)

            eig_values, eig_vectors = np.linalg.eigh(covariance_matrix)
            variation = eig_values[0] / (eig_values[0] + eig_values[1])

            normal_vector = eig_vectors[:, 0]
            self.normals[i] = Normal(normal_vector[0], normal_vector[1], variation)

            if i > self.half_size + 1:
                prev_normal = self.normals[i - 1]
                if np.dot([prev_normal.x, prev_normal.y], [self.normals[i].x, self.normals[i].y]) < 0:
                    self.normals[i].x = -self.normals[i].x
                    self.normals[i].y = -self.normals[i].y

        for i in reversed(range(self.half_size)):
            self.normals[i].x = 2 * self.normals[i + 1].x - self.normals[i + 2].x
            self.normals[i].y = 2 * self.normals[i + 1].y - self.normals[i + 2].y
            # Note: Variation is theoretically non-negative, but linear extrapolation may produce negative values at the boundaries.
            self.normals[i].variation = 2 * self.normals[i + 1].variation - self.normals[i + 2].variation
            self.normals[-i - 1].x = 2 * self.normals[-i - 2].x - self.normals[-i - 3].x
            self.normals[-i - 1].y = 2 * self.normals[-i - 2].y - self.normals[-i - 3].y
            self.normals[-i - 1].variation = 2 * self.normals[-i - 2].variation - self.normals[-i - 3].variation

        return self.normals

    def compute_variations(self) -> None:
        """
        Compute and cache the curve variations.
        """
        if self.features.get(FeatureType.VARIATION) is not None:
            return

        self.compute_normals()
        self.features[FeatureType.VARIATION] = np.array([normal.variation for normal in self.normals])

    def compute_direct_angles(self) -> None:
        """
        [Deprecated] Compute tangent angles using only immediate left and right neighbors.
        """
        if self.features.get(FeatureType.DIRECT_ANGLE) is not None:
            return

        angles = np.zeros(self.num_points)

        for i in range(1, self.num_points - 1):
            angles[i] = 1 / 2 * np.arctan2(self.points[i + 1, 1] - self.points[i, 1], self.points[i + 1, 0] - self.points[i, 0]) + 1 / 2 * np.arctan2(self.points[i, 1] - self.points[i - 1, 1], self.points[i, 0] - self.points[i - 1, 0])

        angles[0] = 2 * angles[1] - angles[2]
        angles[-1] = 2 * angles[-2] - angles[-3]

        self.features[FeatureType.DIRECT_ANGLE] = angles

    def compute_direct_angle_diffs(self) -> None:
        """
        [Deprecated] Compute the numerical derivative (difference) of direct angles.
        """
        if self.features.get(FeatureType.DIRECT_ANGLE_DIFF) is not None:
            return

        self.compute_direct_angles()
        angles = self.features.get(FeatureType.DIRECT_ANGLE)
        self.features[FeatureType.DIRECT_ANGLE_DIFF] = self.compute_value_diffs(angles)

    def compute_angles(self) -> None:
        """
        Compute tangent angles derived from estimated normals using neighborhood PCA.
        """
        if self.features.get(FeatureType.ANGLE) is not None:
            return

        self.compute_normals()
        angles = np.zeros(self.num_points)
        step_pos = []
        for i in range(self.num_points):
            normal = self.normals[i]
            tangent_vector = np.array([-normal.y, normal.x])
            if i == 0:
                forward_vector = self.points[2] - self.points[0]
            elif i == self.num_points - 1:
                forward_vector = self.points[-1] - self.points[-3]
            else:
                forward_vector = self.points[i + 1] - self.points[i - 1]

            if np.dot(tangent_vector, forward_vector) < 0:
                tangent_vector = -tangent_vector

            angles[i] = np.arctan2(tangent_vector[1], tangent_vector[0])
            # Record discontinuity positions (phase jumps)
            if i > 0 and np.abs(angles[i] - angles[i - 1]) > np.pi:
                step_pos.append(i)

        # Perform phase unwrapping to ensure continuous angle values
        if step_pos:
            for pos in step_pos:
                if angles[pos - 1] < angles[pos]:
                    offset = -2 * np.pi
                else:
                    offset = 2 * np.pi

                angles[pos:] += offset

        self.features[FeatureType.ANGLE] = angles

    def compute_angle_diffs(self) -> None:
        """
        Compute and cache the numerical derivative (difference) of the tangent angles.
        """
        if self.features.get(FeatureType.ANGLE_DIFF) is not None:
            return

        self.compute_angles()
        angles = self.features.get(FeatureType.ANGLE)
        self.features[FeatureType.ANGLE_DIFF] = self.compute_value_diffs(angles)

    @staticmethod
    def compute_value_diffs(values: np.ndarray, jump: int = 1) -> np.ndarray:
        """
        Compute central differences for a sequence of values.
        :param values: Input array of values.
        :param jump: The step size for computing the difference.
        :return: An array of numerical derivatives.
        """
        n = len(values)
        diffs = np.zeros(n)

        for i in range(jump, n - jump):
            diffs[i] = (values[i + jump] - values[i - jump]) / (2 * jump)

        for i in range(jump):
            diffs[i] = (values[i + jump] - values[0]) / (jump + i)
            diffs[-i - 1] = (values[-1] - values[-i - 1 - jump]) / (jump + i)

        return diffs
