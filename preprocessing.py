import numpy as np
from enums import FilterType
from typing import List, Optional, Any


class Preprocessing:
    @staticmethod
    def filter_points(points: np.ndarray, filter_type: FilterType, half_size: int,
                      normals: Optional[List[Any]] = None, sigma: float = 1.0,
                      k_min: float = 0.3, k_max: float = 3.0, sigma_n: float = 0.2) -> np.ndarray:
        """
        Apply various filtering techniques to smooth the 2D point cloud.
        :param points: Input (n, 2) numpy array.
        :param filter_type: Type of the filter.
        :param half_size: Radius of the filtering window.
        :param normals: Optional list of Normal objects (required for adaptive/bilateral filters).
        :param sigma: Standard deviation for the spatial Gaussian kernel.
        :param k_min: Minimum multiplier for adaptive sigma.
        :param k_max: Maximum multiplier for adaptive sigma.
        :param sigma_n: Standard deviation for the normal-based weight in bilateral filtering.
        :return: Filtered point cloud (n, 2) numpy array.
        """
        if filter_type == FilterType.DISABLE:
            return points
        elif filter_type == FilterType.MEAN:
            return Preprocessing.mean_filter_points(points, half_size)
        elif filter_type == FilterType.GAUSSIAN:
            return Preprocessing.gaussian_filter_points(points, half_size, sigma)
        elif filter_type == FilterType.ADAPTIVE_GAUSSIAN:
            if normals is None:
                raise ValueError("Normals must be provided for adaptive Gaussian filtering.")
            return Preprocessing.adaptive_gaussian_filter_points(points, normals, half_size, sigma, k_min, k_max)
        elif filter_type == FilterType.BILATERAL:
            if normals is None:
                raise ValueError("Normals must be provided for bilateral filtering.")
            return Preprocessing.bilateral_filter_points(points, normals, half_size, sigma_d=sigma, sigma_n=sigma_n)
        elif filter_type == FilterType.GUIDED:
            return Preprocessing.guided_filter_points(points, half_size)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    @staticmethod
    def mean_filter_points(points: np.ndarray, half_size: int) -> np.ndarray:
        n = len(points)
        filtered_points = np.zeros_like(points)
        for i in range(0, n):
            current_half_size = min(i, n - 1 - i, half_size)
            start_idx = max(0, i - current_half_size)
            end_idx = min(n, i + current_half_size + 1)
            neighbor_points = points[start_idx:end_idx]
            filtered_points[i] = np.mean(neighbor_points, axis=0)

        return filtered_points

    @staticmethod
    def gaussian_filter_points(points: np.ndarray, half_size: int, sigma: float = 1.0) -> np.ndarray:
        n = len(points)
        sigma_sqr = sigma * sigma
        filtered_points = np.zeros_like(points)

        for i in range(n):
            start_idx = max(0, i - half_size)
            end_idx = min(n, i + half_size + 1)

            neighbor_points = points[start_idx:end_idx]
            current_point = points[i]
            sqr_distances = np.sum((neighbor_points - current_point) ** 2, axis=1)

            weights = np.exp(-0.5 * sqr_distances / sigma_sqr)
            total_weight = np.sum(weights)

            if total_weight > 0:
                weights = weights / total_weight
                filtered_points[i] = np.sum(neighbor_points * weights[:, np.newaxis], axis=0)
            else:
                filtered_points[i] = points[i]

        return filtered_points

    @staticmethod
    def adaptive_gaussian_filter_points(points: np.ndarray, normals: List[Any],
                                        half_size: int, base_sigma: float,
                                        k_min: float, k_max: float) -> np.ndarray:
        n = len(points)
        filtered_points = np.zeros_like(points)

        curvature = np.array([normal.variation for normal in normals])
        # Normalize curvature to [0, 1] range
        if np.max(curvature) > 0:
            normalized_curvature = curvature / np.max(curvature)
        else:
            normalized_curvature = np.zeros_like(curvature)

        # Compute adaptive sigma based on normalized curvature
        k = k_max - (k_max - k_min) * normalized_curvature
        adaptive_sigma = base_sigma * k

        for i in range(n):
            start_idx = max(0, i - half_size)
            end_idx = min(n, i + half_size + 1)

            neighbor_points = points[start_idx:end_idx]
            current_point = points[i]
            sqr_distances = np.sum((neighbor_points - current_point) ** 2, axis=1)

            sigma_sqr = adaptive_sigma[i] * adaptive_sigma[i]
            weights = np.exp(-0.5 * sqr_distances / sigma_sqr)
            total_weight = np.sum(weights)

            if total_weight > 0:
                weights = weights / total_weight
                filtered_points[i] = np.sum(neighbor_points * weights[:, np.newaxis], axis=0)
            else:
                filtered_points[i] = points[i]

        return filtered_points

    @staticmethod
    def bilateral_filter_points(points: np.ndarray, normals: List[Any], half_size: int,
                                sigma_d: float = 1.0, sigma_n: float = 0.2) -> np.ndarray:
        n = len(points)
        sigma_d_sqr = sigma_d * sigma_d
        sigma_n_sqr = sigma_n * sigma_n
        filtered_points = np.zeros_like(points)

        for i in range(n):
            start_idx = max(0, i - half_size)
            end_idx = min(n, i + half_size + 1)

            current_point = points[i]
            current_normal = np.array([normals[i].x, normals[i].y])
            neighbor_points = points[start_idx:end_idx]

            # Compute the spatial distances
            diff_vectors = neighbor_points - current_point
            sqr_distances = np.sum(diff_vectors ** 2, axis=1)
            w_d = np.exp(-0.5 * sqr_distances / sigma_d_sqr)

            # Compute the projection of the difference vectors onto the normal vector
            projections = np.dot(diff_vectors, current_normal)
            w_n = np.exp(-0.5 * projections ** 2 / sigma_n_sqr)

            # Combined bilateral weight
            weights = w_d * w_n
            total_weight = np.sum(weights)

            if total_weight > 0:
                weights = weights / total_weight
                filtered_points[i] = np.sum(neighbor_points * weights[:, np.newaxis], axis=0)
            else:
                filtered_points[i] = points[i]

        return filtered_points

    @staticmethod
    def guided_filter_points(points, half_size, eps=1e-1, use_mean=False):
        """[Deprecated] Guided filter implementation for point cloud smoothing."""
        n = len(points)
        if use_mean:
            # Precalculate a and b for each point
            a_values = np.zeros(n)
            b_values = np.zeros((n, 2))

            # First pass: compute local linear model parameters for each neighborhood
            for i in range(n):
                start_idx = max(0, i - half_size)
                end_idx = min(n, i + half_size + 1)
                neighbor_points = points[start_idx:end_idx]

                p_mean = np.mean(neighbor_points, axis=0)
                variance = np.mean(np.sum(neighbor_points * neighbor_points, axis=1)) - np.sum(p_mean * p_mean)

                a_values[i] = variance / (variance + eps)
                b_values[i] = p_mean - a_values[i] * p_mean

            # Second pass: aggregate parameters to compute final filtered coordinates
            filtered_points = np.zeros_like(points)
            count = np.zeros(n)

            for i in range(n):
                start_idx = max(0, i - half_size)
                end_idx = min(n, i + half_size + 1)

                for j in range(start_idx, end_idx):
                    filtered_points[i] += a_values[j] * points[i] + b_values[j]
                    count[i] += 1

            filtered_points /= count[:, np.newaxis]

        else:
            # Direct use each point's neighborhood to compute the result
            filtered_points = np.zeros_like(points)

            for i in range(n):
                start_idx = max(0, i - half_size)
                end_idx = min(n, i + half_size + 1)
                neighbor_points = points[start_idx:end_idx]

                p_mean = np.mean(neighbor_points, axis=0)
                variance = np.mean(np.sum(neighbor_points * neighbor_points, axis=1)) - np.sum(p_mean * p_mean)

                a_i = variance / (variance + eps)
                b_i = p_mean - a_i * p_mean
                filtered_points[i] = a_i * points[i] + b_i

        return filtered_points
