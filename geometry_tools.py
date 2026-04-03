import numpy as np
from scipy.optimize import least_squares

class GeometryTools:
    @staticmethod
    def fit_line(points: np.ndarray):
        """
        Perform linear least-squares fitting on 2D points.
        :param points: (n, 2) numpy array of 2D points.
        :return: Tuple of (slope k, intercept b).
        """
        x = points[:, 0]
        y = points[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return k, b

    @staticmethod
    def line_fit_error(points: np.ndarray) -> float:
        """
        Calculate the sum of suared residuals for a fitted line.
        :param points: (n, 2) numpy array of 2D points.
        :return: Total squared residual error.
        """
        x = points[:, 0]
        y = points[:, 1]
        k, b = GeometryTools.fit_line(points)
        residuals = y - (k * x + b)
        return np.sum(residuals ** 2)

    @staticmethod
    def get_projection_point(point: np.ndarray, k: float, b: float) -> np.ndarray:
        """
        Calculate the projection of a point onto a line defined by y = kx + b.
        :param point: 2D coordinates (x, y).
        :param k: Slope of the line.
        :param b: Intercept of the line.
        :return: Projected 2D coordinates (x', y').
        """
        x, y = point
        x_proj = (k * (y - b) + x) / (k ** 2 + 1)
        y_proj = k * x_proj + b
        return np.array([x_proj, y_proj])

    @staticmethod
    def get_projection_point2(point: np.ndarray, line_point1: np.ndarray, line_vec: np.ndarray) -> np.ndarray:
        """
        Calculate the orthogonal projection of a point onto a line defined by a point and a direction vector.
        :param point: 2D coordinates (x, y).
        :param line_point1: A point on the line.
        :param line_vec: Direction vector (dx, dy).
        :return: Projected 2D coordinates (x', y').
        """
        x, y = point
        x1, y1 = line_point1
        dx, dy = line_vec

        # Calculate projection coefficient t
        t = ((x - x1) * dx + (y - y1) * dy) / (dx ** 2 + dy ** 2)
        x_prime = x1 + t * dx
        y_prime = y1 + t * dy
        return np.array([x_prime, y_prime])

    @staticmethod
    def fit_circle(points: np.ndarray):
        """
        Perform circular fitting using least squares optimization.
        :param points: (n, 2) numpy array of 2D points.
        :return: Tuple of (center_coordinates, radius, start_angle, end_angle).
        """
        def circle_residuals(params):
            xc, yc, r = params
            return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2) - r

        x_m = np.mean(points[:, 0])
        y_m = np.mean(points[:, 1])
        r_m = np.mean(np.sqrt((points[:, 0] - x_m) ** 2 + (points[:, 1] - y_m) ** 2))

        result = least_squares(circle_residuals, [x_m, y_m, r_m])
        circle_center = np.array([result.x[0], result.x[1]])
        radius = result.x[2]

        start_angle = np.arctan2(points[0, 1] - circle_center[1], points[0, 0] - circle_center[0])
        end_angle = np.arctan2(points[-1, 1] - circle_center[1], points[-1, 0] - circle_center[0])

        return circle_center, radius, start_angle, end_angle

    @staticmethod
    def circle_fit_error(points):
        """
        Calculate the sum of suared residuals for a fitted circle.
        :param points: (n, 2) numpy array of 2D points.
        :return: Total squared residual error.
        """
        def circle_residuals(params):
            xc, yc, r = params
            return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2) - r

        x_m = np.mean(points[:, 0])
        y_m = np.mean(points[:, 1])
        r_m = np.mean(np.sqrt((points[:, 0] - x_m) ** 2 + (points[:, 1] - y_m) ** 2))

        result = least_squares(circle_residuals, [x_m, y_m, r_m])
        # scipy's least_squares returns 0.5 * sum(residuals^2)
        return 2 * result.cost
