import numpy as np
from single_side import SingleSide
from segment_info import SegmentInfo
from plotter import Plotter
from enums import MeasuringMethod
from typing import Optional, Tuple, Any


class FlushGapTool:
    def __init__(self, part1_points: np.ndarray, part2_points: np.ndarray,
                 part1_info: SegmentInfo, part2_info: SegmentInfo,
                 section_name: str = None):
        """
        Measurement tool for gap and flush between door and fender profiles.
        Note: Each profile is assumed to contain two tangent points.
        :param part1_points: Point cloud for the first part
        :param part2_points: Point cloud for the second part
        :param part1_info: Segmentation metadata for the first part.
        :param part2_info: Segmentation metadata for the second part.
        :param section_name: Optional identifier for the cross-section.
        """
        self.part1_points = part1_points
        self.part2_points = part2_points
        self.part1_info = part1_info
        self.part2_info = part2_info
        self.part1_side = SingleSide(self.part1_points, self.part1_info)
        self.part2_side = SingleSide(self.part2_points, self.part2_info)
        self.section_name = section_name

    def calculate_flush_gap(self, method: MeasuringMethod, ref_direction: Optional[np.ndarray] = None,
                            get_pt_info: bool = False, plot_result: bool = True) -> Any:
        """
        Main execution entry to calculate flush and gap values.
        :param method: The specific measurement protocol to use.
        :param ref_direction: Global reference vector (required for GLOBAL method).
        :param get_pt_info: If True, returns detailed KDP metadata.
        :param plot_result: Whether to draw the measurement results.
        :return: (flush, gap) or (flush, gap, measure_info_dict).
        """
        if method == MeasuringMethod.GLOBAL and ref_direction is None:
            raise ValueError("Global vector must be provided.")

        if plot_result:
            if method != MeasuringMethod.PRIMARY2:
                Plotter.plot_flush_gap_preparing(self.part1_side, self.part2_side, section_name=self.section_name, wait_for=True)
            else:
                Plotter.plot_flush_gap_preparing(self.part2_side, self.part1_side, section_name=self.section_name, wait_for=True)

        flush, flush_idx1, flush_idx2, flush_pt1, flush_pt2, flush_dir = self._calculate_flush(method, ref_direction)
        gap, gap_idx1, gap_idx2, gap_pt1, gap_pt2, gap_dir = self._calculate_gap(method, ref_direction)

        if plot_result:
            Plotter.plot_gap_flush_submit()

        if not get_pt_info:
            return flush, gap
        else:
            measure_info = {
                "flush_idx1": flush_idx1,
                "flush_idx2": flush_idx2,
                "flush_pt1": flush_pt1,
                "flush_pt2": flush_pt2,
                "flush_dir": flush_dir,
                "gap_idx1": gap_idx1,
                "gap_idx2": gap_idx2,
                "gap_pt1": gap_pt1,
                "gap_pt2": gap_pt2,
                "gap_dir": gap_dir
            }
            return flush, gap, measure_info

    def _calculate_flush(self, method: MeasuringMethod, ref_direction: Optional[np.ndarray] = None,
                         plot_result: bool = True) -> Tuple:
        """
        Calculate the flush value based on the specified measurement method.
        """
        p1 = self.part1_side.tangent_projection  # KDP 1
        p2 = self.part2_side.tangent_projection  # KDP 2
        k1, b1 = self.part1_side.k, self.part1_side.b
        k2, b2 = self.part2_side.k, self.part2_side.b

        if method == MeasuringMethod.PRIMARY1:
            flush = (k1 * p2[0] - p2[1] + b1) / np.sqrt(k1 ** 2 + 1)
            flush_dir = (-k1, 1)
            param = k1
        elif method == MeasuringMethod.PRIMARY2:
            flush = (k2 * p1[0] - p1[1] + b2) / np.sqrt(k2 ** 2 + 1)
            flush_dir = (-k2, 1)
            param = k2
        elif method == MeasuringMethod.BISECTOR:
            k = (k1 + k2) / 2
            flush = ((k * p2[0] - p2[1]) - (k * p1[0] - p1[1])) / np.sqrt(k ** 2 + 1)
            flush_dir = (-k, 1)
            param = k
        elif method == MeasuringMethod.GLOBAL:
            measure_vec_origin = np.array(ref_direction)
            flush_dir = measure_vec_origin / np.linalg.norm(measure_vec_origin)
            point_vec = np.array(p1) - np.array(p2)
            flush = (np.dot(point_vec, ref_direction))
            param = flush_dir
        else:
            raise ValueError("Invalid measuring direction")

        if plot_result:
            Plotter.plot_flush(method, p1, p2, param, flush)

        return flush, self.part1_side.tangent_idx, self.part2_side.tangent_idx, p1, p2, flush_dir

    def _calculate_gap(self, method: MeasuringMethod, ref_direction: Optional[np.ndarray] = None,
                       plot_result: bool = True) -> Tuple:
        """
        Calculate the gap value based on the specified measurement method.
        """
        p1 = self.part1_side.tangent_projection
        p2 = self.part2_side.tangent_projection
        k1 = self.part1_side.k
        k2 = self.part2_side.k
        arc_range1 = self.part1_side.segment.tangents
        arc_range2 = self.part2_side.segment.tangents

        p_ref = (p1 + p2) / 2
        if method == MeasuringMethod.PRIMARY1:
            gap_dir = (1, k1)
        elif method == MeasuringMethod.PRIMARY2:
            gap_dir = (1, k2)
        elif method == MeasuringMethod.BISECTOR:
            k = (k1 + k2) / 2
            gap_dir = (1, k)
        elif method == MeasuringMethod.GLOBAL:
            gap_dir = (-ref_direction[1], ref_direction[0])  # Perpendicular to the global measuring direction
        else:
            raise ValueError("Invalid measuring direction")

        idx1, b_pos1 = self.find_support_point_with_reference(
            self.part1_side.points, (arc_range1[0], arc_range1[1]), gap_dir, p_ref
        )
        idx2, b_pos2 = self.find_support_point_with_reference(
            self.part2_side.points, (arc_range2[0], arc_range2[1]), gap_dir, p_ref
        )
        gap = np.abs(b_pos1 - b_pos2)

        gap_pt1 = self.part1_side.points[idx1]
        gap_pt2 = self.part2_side.points[idx2]
        if plot_result:
            Plotter.plot_gap(method, p1, p2, gap_dir, gap_pt1, gap_pt2, gap)

        return gap, idx1, idx2, gap_pt1, gap_pt2, gap_dir

    @staticmethod
    def find_support_point_with_reference(points: np.ndarray, arc_range: Tuple[int, int],
                                          gap_dir: Tuple[float, float], reference_point: np.ndarray) -> Tuple[int, float]:
        """
        Identify the supporting point by finding the geometric contact position of a sweep line on the mating segment.
        :param points: Full point cloud array.
        :param arc_range: Start and end indices defining the mating segment (arc region).
        :param gap_dir: Gap measuring direction (sweep direction).
        :param reference_point: Anchor point used to construct the initial reference sweep line.
        :return: A tuple containing (support_point_index, line_intercept_b).
        """
        start, end = arc_range
        arc_points = points[start:end + 1]

        n = np.array(gap_dir, dtype=np.float64)
        n = n / np.linalg.norm(n)

        # Construct the reference line Ax + By + C = 0 through p_ref
        x0, y0 = reference_point
        C = -n[0] * x0 - n[1] * y0

        # Calculate signed distances to find the point closest to the sweep line
        signed_dists = arc_points @ n + C

        # Check if the reference line intersects the point set (mixed signs)
        has_positive = np.any(signed_dists > 0)
        has_negative = np.any(signed_dists < 0)

        if has_positive and has_negative:
            # Intersection case: find the closest boundaries from both sides
            max_pos_idx = np.argmax(signed_dists)
            min_neg_idx = np.argmin(signed_dists)

            # Select the side with the smaller absolute displacement to p_ref
            if signed_dists[max_pos_idx] < abs(signed_dists[min_neg_idx]):
                min_idx_local = max_pos_idx
            else:
                min_idx_local = min_neg_idx
        else:
            # Single-side case: find the point with minimum absolute distance
            min_idx_local = np.argmin(np.abs(signed_dists))

        min_idx_global = start + min_idx_local

        support_point = points[min_idx_global]
        # Find b such that n[0] * x + n[1] * y + b = 0
        b = -n[0] * support_point[0] - n[1] * support_point[1]

        return min_idx_global, b
