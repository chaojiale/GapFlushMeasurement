import numpy as np
from segment_info import SegmentInfo
from geometry_tools import GeometryTools
from typing import Dict, Tuple, Optional


class Optimizer:
    def __init__(self, filtered_points: np.ndarray):
        """
        Initialize the breakpoint optimizer.
        :param filtered_points: Preprocessed point cloud array.
        """
        self.points = filtered_points
        self.split_error_cache: Dict[Tuple[int, int], float] = {}

    def _calculate_split_error(self, current_pos: int, prev_split: int, next_split: int,
                               refer_two_segment_types: Optional[str] = None) -> float:
        """
        Calculate the total fitting error (mean squared residual) at a potential breakpoint.
        :param current_pos: Index of the candidate breakpoint.
        :param prev_split: Index of the preceding breakpoint.
        :param next_split: Index of the succeeding breakpoint.
        :param refer_two_segment_types: Geometric types ('l' for line, 'a' for arc) of adjacent segments.
        :return: Combined mean fitting error of the left and right segments.
        """
        if refer_two_segment_types is not None and len(refer_two_segment_types) != 2 and any(c not in 'la' for c in refer_two_segment_types):
            raise ValueError("refer_all_segment_types must be 'l' or 'a'")

        def segment_error(segment_points, segment_type):
            if segment_type == 'l':
                return GeometryTools.line_fit_error(segment_points)
            elif segment_type == 'a':
                return GeometryTools.circle_fit_error(segment_points)
            else:
                e_line = GeometryTools.line_fit_error(segment_points)
                e_circle = GeometryTools.circle_fit_error(segment_points)
                return min(e_line, e_circle)

        # Calculate mean error for the left segment (prev_split to current_pos)
        if (prev_split, current_pos) in self.split_error_cache:
            e_left_mean = self.split_error_cache[(prev_split, current_pos)]
        else:
            left_points = self.points[prev_split:current_pos + 1]
            e_left = segment_error(left_points, refer_two_segment_types[0] if refer_two_segment_types else None)
            e_left_mean = e_left / len(left_points)
            self.split_error_cache[(prev_split, current_pos)] = e_left_mean

        # Calculate mean error for the right segment (current_pos to next_split)
        if (current_pos, next_split) in self.split_error_cache:
            e_right_mean = self.split_error_cache[(current_pos, next_split)]
        else:
            right_points = self.points[current_pos:next_split + 1]
            e_right = segment_error(right_points, refer_two_segment_types[1] if refer_two_segment_types else None)
            e_right_mean = e_right / len(right_points)
            self.split_error_cache[(current_pos, next_split)] = e_right_mean

        return e_left_mean + e_right_mean

    def optimize_breakpoints(self, initial_info: SegmentInfo, window: int = 5,
                             max_iters: int = 3, segment_offset: int = 5) -> SegmentInfo:
        """
        Refine the positions of initial breakpoints by minimizing local fitting residuals.
        :param initial_info: Initial breakpoint detection results.
        :param window: Search range radius.
        :param max_iters: Maximum number of optimization iterations.
        :param segment_offset: Offset points used to mitigate inaccuracies from adjacent breakpoints.
        :return: A SegmentInfo object with optimized corner and tangent indices.
        """
        breakpoints = sorted(
            [(pos, 'c') for pos in initial_info.corners] +
            [(pos, 't') for pos in initial_info.tangents],
            key=lambda x: x[0]
        )

        key_num = len(breakpoints)
        refer_all_segment_types = initial_info.line_types
        if refer_all_segment_types is not None and len(refer_all_segment_types) != key_num + 1 and any(c not in 'la' for c in refer_all_segment_types):
            raise ValueError("refer_all_segment_types must be 'l' or 'a'")

        optimized = breakpoints.copy()

        for epoch in range(max_iters):
            break_out = False

            if epoch == 0:
                phases = [range(key_num), range(key_num - 2, -1, -1)]
            else:
                phases = [range(1, key_num), range(key_num - 2, -1, -1)]

            for phase in phases:
                boundary_flag = False
                for i in phase:
                    current_pos, current_type = optimized[i]
                    prev_split = (optimized[i - 1][0] if i > 0 else 0) + segment_offset
                    next_split = (optimized[i + 1][0] if i < key_num - 1 else len(self.points) - 1) - segment_offset

                    refer_two_segment_types = None
                    if refer_all_segment_types is not None:
                        refer_two_segment_types = refer_all_segment_types[i:i + 2]

                    candidates = range(current_pos - window, current_pos + window + 1)
                    # Search for the optimal position
                    min_error = self._calculate_split_error(current_pos, prev_split, next_split, refer_two_segment_types)
                    current_error = min_error
                    best_pos = current_pos
                    for candidate in candidates:
                        if candidate == current_pos or candidate < prev_split or candidate > next_split:
                            continue

                        error = self._calculate_split_error(candidate, prev_split, next_split, refer_two_segment_types)
                        if error < min_error and (current_error - error) / current_error > 0.01:
                            min_error = error
                            best_pos = candidate

                    # Update breakpoint position
                    if best_pos != current_pos:
                        optimized[i] = (best_pos, current_type)
                        if best_pos == current_pos - window or best_pos == current_pos + window:
                            boundary_flag = True

                if not boundary_flag:
                    break_out = True
                    break

            if break_out:
                break

        # Separate optimized indices back into corners and tangents
        corners = [pos for pos, t in optimized if t == 'c']
        tangents = [pos for pos, t in optimized if t == 't']
        return SegmentInfo(corners, tangents)
