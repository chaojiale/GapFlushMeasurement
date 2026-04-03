import numpy as np
from point_recognizer import PointRecognizer
from plotter import Plotter
from preprocessing import Preprocessing, FilterType
from feature_analysis import FeatureType, FeatureAnalysis
from optimizer import Optimizer
from typing import Tuple, Optional

from segment_info import SegmentInfo


class BreakpointDetection:
    def __init__(self, points: np.ndarray, corner_count: int = -1,
                 tangent_count: int = -1, line_types: Optional[str] = None):
        """
        Initialize the breakpoint detection handler.
        :param points: Input 2D point cloud.
        :param corner_count: Expected number of corner points (-1 if unknown).
        :param tangent_count: Expected number of tangent points (-1 if unknown).
        :param line_types: Geometric primitives expected in the profile.
        """
        self.points = points
        self.corner_count = corner_count
        self.tangent_count = tangent_count
        self.line_types = line_types

    def Execute(self, filter_type: FilterType, half_size: int, sigma: float,
                param: float = 1.0, ignore_edge: int = 10,
                refer_info: Optional[SegmentInfo] = None, show_log: bool = True,
                plot_result: bool = False, name: str = "") -> Tuple[SegmentInfo, SegmentInfo]:
        """
        Execute the full pipeline for breakpoint detection, including filtering, initial detection, and optimization.
        :param filter_type: Type of filter to apply.
        :param half_size: Neighborhood radius for feature analysis and filtering.
        :param sigma: Standard deviation for the Gaussian kernel.
        :param param: Scaling factor for the suppression radius. Values >= 1.0 are recommended;
                      however, values < 1.0 are acceptable if the number of corners and tangents
                      is known and the distances between breakpoints are small.
        :param ignore_edge: Number of points to ignore at the boundaries.
        :param refer_info: Reference info for visualization.
        :param show_log: Whether to print detection results.
        :param plot_result: Whether to generate visualization plots.
        :param name: Identifier string for plot titles.
        :return: A tuple containing (initial_segment_info, optimized_segment_info).
        """
        # Step 1: Preliminary feature analysis to compute normals and variations
        analysis = FeatureAnalysis(self.points, half_size)

        # Step 2: Apply the selected filter to the point cloud
        normals = None
        if filter_type in [FilterType.ADAPTIVE_GAUSSIAN, FilterType.BILATERAL]:
            normals = analysis.compute_normals()

        analysis.compute_variations()
        curvatures = analysis.features.get(FeatureType.VARIATION)
        filtered_points = Preprocessing.filter_points(self.points, filter_type, half_size, normals, sigma, 0.3, 3, sigma_n=sigma * 0.2)

        # Step 3: Compute tangent angles from the filtered points
        current_analysis = FeatureAnalysis(filtered_points, half_size)
        current_analysis.compute_angles()
        angles = current_analysis.features.get(FeatureType.ANGLE)

        if plot_result:
            fig_title = f'{name} {filter_type}'
            Plotter.plot_scatter_points(filtered_points, refer_info, f'Filtered points after {fig_title} filter')
            Plotter.plot_curve_with_breakpoints(angles, refer_info, title=f"Angles {fig_title}")

        # Step 4: Perform initial breakpoint detection
        init_info = PointRecognizer.detect_breakpoints_manual(angles, curvatures, sigma=sigma, neighbor_radius=half_size, expected_corners=self.corner_count, expected_tangents=self.tangent_count, param=param, ignore_edge=ignore_edge, plot_result=plot_result)

        if show_log:
            print(f"Detected results before optimization: {init_info}")

        # Step 5: Refine the detected breakpoint positions
        optimizer = Optimizer(filtered_points)
        init_info.line_types = self.line_types
        optimized_info = optimizer.optimize_breakpoints(init_info, window=5)

        if show_log:
            print(f"Detected results after optimization: {optimized_info}")

        if plot_result:
            Plotter.plot_scatter_points(self.points, optimized_info, title=f"Detected breakpoints of {fig_title}")

        return init_info, optimized_info
