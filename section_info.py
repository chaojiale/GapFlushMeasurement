import numpy as np
from scipy.spatial.transform import Rotation
import os
from typing import Optional, Tuple


class SectionInfo:
    def __init__(self, section_name: str, save_file_path: str, part1_name: str = "Door", part2_name: str = "Fender"):
        """
        Initialize section info and load point clouds.
        :param section_name: Name of the section.
        :param save_file_path: Path to the point cloud file.
        :param part1_name: Identifier for the first part.
        :param part2_name: Identifier for the second part.
        """
        self.section_name = section_name
        self.save_file_path = save_file_path

        # Point cloud data (nx3 numpy arrays)
        self.part1_points: np.ndarray = np.empty((0, 3))
        self.part2_points: np.ndarray = np.empty((0, 3))

        # Plane normal vector
        self.plane_normal: Optional[np.ndarray] = None
        self.has_get_plane_normal = False

        # Rotation matrices
        self.rotation_matrix: Optional[np.ndarray] = None
        self.inverse_rotation_matrix: Optional[np.ndarray] = None

        # Transformed point clouds (aligned to XOY plane)
        self.rotated_part1_points: Optional[np.ndarray] = None
        self.rotated_part2_points: Optional[np.ndarray] = None
        self.has_create_rotated_cloud = False

        # Load data
        if os.path.exists(save_file_path):
            self.part1_points, self.part2_points = self.load_gap_flush_file(save_file_path, part1_name, part2_name)

    @staticmethod
    def load_point_cloud_txt(file_path: str) -> np.ndarray:
        """
        Load 2D coordinates from a text file.
        :param file_path: Path to the source file.
        :return: xy_points as an (n, 2) numpy array.
        """
        data = np.loadtxt(file_path, delimiter=",", comments=None, dtype=np.float32)
        xy_points = data[:, :2]
        return xy_points

    @staticmethod
    def save_point_cloud_text(points: np.ndarray, file_path: str) -> None:
        """
        Save points to a file.
        :param points: Input points (n, 2) or (n, 3).
        :param file_path: Target file path.
        """
        if points.ndim == 2 and points.shape[1] == 2:
            points = np.hstack((points, np.zeros((points.shape[0], 1), dtype=points.dtype)))

        np.savetxt(file_path, points, fmt="%.6f", delimiter=",")

    @staticmethod
    def load_gap_flush_file(file_path: str, part1_name: str, part2_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse dual point clouds from a formatted file.
        :param file_path: Source file path.
        :param part1_name: Identifier for the first cloud part.
        :param part2_name: Identifier for the second cloud part.
        :return: Tuple of two (n, 3) numpy arrays.
        """
        try:
            with open(file_path, 'r') as in_file:
                cloud1_points = []
                cloud2_points = []
                current_cloud = None

                for line in in_file:
                    line = line.strip()
                    if not line:
                        continue

                    if line[0] == '#':
                        if part1_name in line:
                            current_cloud = 'cloud1'
                        elif part2_name in line:
                            current_cloud = 'cloud2'
                        else:
                            current_cloud = None
                        continue

                    if current_cloud:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            try:
                                x = float(parts[0])
                                y = float(parts[1])
                                z = float(parts[2])

                                if current_cloud == 'cloud1':
                                    cloud1_points.append([x, y, z])
                                elif current_cloud == 'cloud2':
                                    cloud2_points.append([x, y, z])
                            except ValueError:
                                raise ValueError(f"Invalid point data: {line}")

                # To numpy arrays
                cloud1 = np.array(cloud1_points) if cloud1_points else np.empty((0, 3))
                cloud2 = np.array(cloud2_points) if cloud2_points else np.empty((0, 3))

                return cloud1, cloud2

        except IOError:
            raise IOError(f"Could not read file: {file_path}")

    def initialize_rotated_point_cloud(self) -> bool:
        """
        Initialize the coordinate transformation by aligning the plane normal to the Z-axis.

        """
        if not self.has_get_plane_normal and not self.try_get_plane_normal():
            return False

        initial_normal = self.plane_normal
        target_normal = np.array([0, 0, 1])

        # Compute the rotation matrix, using the method equivalent to Eigen's FromTwoVectors
        rotation = Rotation.align_vectors(
            [target_normal],
            [initial_normal]
        )[0]

        self.rotation_matrix = rotation.as_matrix()
        self.inverse_rotation_matrix = rotation.inv().as_matrix()

        # Transform points using the rotation matrix
        def transform_points(points):
            if points.size == 0:
                return np.empty((0, 3))

            # Equal to PCL's transformPointCloud
            return (self.rotation_matrix @ points.T).T

        self.rotated_part1_points = transform_points(self.part1_points)
        self.rotated_part2_points = transform_points(self.part2_points)

        self.has_create_rotated_cloud = True
        return True

    def try_get_plane_normal(self, bias: float = 0.01, max_tries: int = 10) -> bool:
        """
        Attempt to compute the plane normal vector by randomly sampling three points from the point cloud.
        :param bias: Minimum threshold for the cross product norm.
        :param max_tries: Maximum number of sampling attempts.
        :return: Boolean indicating success.
        """
        if self.part1_points.size == 0 and self.part2_points.size == 0:
            return False

        # Combine the two point clouds
        combined_points = np.vstack([self.part1_points, self.part2_points])
        point_num = combined_points.shape[0]

        if point_num < 3:
            return False

        for _ in range(max_tries):
            idx1, idx2, idx3 = np.random.choice(point_num, 3, replace=False)
            point1 = combined_points[idx1]
            point2 = combined_points[idx2]
            point3 = combined_points[idx3]

            vector1 = point2 - point1
            vector2 = point3 - point1

            cross_vector = np.cross(vector1, vector2)

            if cross_vector[2] < 0:
                cross_vector = -cross_vector

            cross_norm = np.linalg.norm(cross_vector)

            if cross_norm > bias:
                # Normalize
                self.plane_normal = cross_vector / cross_norm
                self.has_get_plane_normal = True
                return True

        return False

    def plane_normal_str(self) -> str:
        return f"({self.plane_normal[0]:.3f},{self.plane_normal[1]:.3f},{self.plane_normal[2]:.3f})"

    def get_transformed_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Transform a 3D vector to the XOY plane coordinate system.
        :param vector: Original normal vector in 3D, should be a unit vector.
        :return: Transformed 2D vector.
        """
        if self.rotation_matrix is None:
            raise ValueError("Rotation matrix is not initialized. Call initialize_rotated_point_cloud() first.")

        # Method 1
        # Apply the rotation matrix to the input vector
        trans_vector = self.rotation_matrix @ vector
        trans_vector[2] = 0
        # Normalize
        length = np.linalg.norm(trans_vector)
        trans_vector /= length

        # # Method 2
        # # Calculate the vertical component using dot product with the plane normal
        # projection_length = np.dot(vector, self.plane_normal)
        # # Subtract the vertical component to get the in-plane projection
        # vector_proj = vector - projection_length * self.plane_normal
        # # Transform the projected vector using the rotation matrix
        # trans_vector2 = self.rotation_matrix @ vector_proj
        # # Normalize
        # length = np.linalg.norm(trans_vector2)
        # trans_vector2 = trans_vector2 / length

        return trans_vector[:2]

    def get_transformed_back_point(self, point: np.ndarray) -> np.ndarray:
        """
        Revert a point from the transformed XOY plane back to the original 3D coordinate system.
        :param point: Point coordinates in the transformed system.
        :return: Point coordinates in the original system.
        """
        if self.inverse_rotation_matrix is None:
            raise ValueError("Inverse rotation matrix is not initialized. Call initialize_rotated_point_cloud() first.")

        original_point = self.inverse_rotation_matrix @ point
        return original_point
