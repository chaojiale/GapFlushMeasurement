import numpy as np
from scipy.ndimage import gaussian_filter1d
from plotter import Plotter
from segment_info import SegmentInfo
from feature_analysis import FeatureAnalysis
from typing import List, Optional


class PointRecognizer:
    @staticmethod
    def manual_peak_detection(signal: np.ndarray, num_peaks: int, neighbor_radius: int,
                              param: float = 1.0, ignore_edge: int = 10,
                              min_thresh: Optional[float] = None, plot_result: bool = False) -> List[int]:
        """
        Perform peak detection using a manual Non-Maximum Suppression (NMS) approach.
        :param signal: 1D input signal array.
        :param num_peaks: Number of expected peaks to detect.
        :param neighbor_radius: Radius for neighborhood suppression.
        :param param: Scaling factor for the suppression radius.
        :param ignore_edge: Number of points to ignore at the boundaries to avoid edge effects.
        :param min_thresh: [Deprecated] Absolute threshold below which peaks are ignored.
        :param plot_result: Whether to visualize the intermediate detection steps.
        :return: Sorted list of peak indices based on signal intensity.
        """
        process_radius = int(neighbor_radius * param)
        abs_signal = np.abs(signal)  # Process absolute values for peak detection
        abs_signal[:ignore_edge] = 0  # Ignore edge points
        abs_signal[-ignore_edge:] = 0
        peaks = []
        if num_peaks == 0:
            return peaks

        have_peak_num = (num_peaks > 0)
        threshold = 0.0
        if not have_peak_num:
            no_zero_values = abs_signal[abs_signal > 0]
            median = float(np.median(no_zero_values))
            mad = float(np.median(np.abs(no_zero_values - median)))
            threshold = median + 8 * mad

        num = 0
        while True:
            num += 1
            if plot_result:
                Plotter.plot_curve_with_breakpoints(abs_signal, segment_info=None, title=f"absolute signal peak detection {num}")

            # Find the current global maximum
            max_idx = int(np.argmax(abs_signal))
            max_val = abs_signal[max_idx]

            # Check threshold
            if min_thresh is not None and max_val < min_thresh:
                abs_signal[max_idx] = 0
                break

            if have_peak_num:
                peaks.append(max_idx)
                if len(peaks) >= num_peaks:
                    break
            else:
                left1 = max(0, max_idx - neighbor_radius - 2)
                left2 = max(0, max_idx - neighbor_radius + 1)
                right1 = min(len(abs_signal) - 1, max_idx + neighbor_radius)
                right2 = min(len(abs_signal) - 1, max_idx + neighbor_radius + 3)

                left_max = abs_signal[left1] if left1 == left2 else np.max(abs_signal[left1:left2])
                right_max = abs_signal[right1] if right1 == right2 else np.max(abs_signal[right1:right2])
                side_max = max(left_max, right_max)

                peak_neighbor_radius = (neighbor_radius + 2) // 3
                # Check if the signal is strictly increasing from max_idx - half_radius to max_idx, and strictly decreasing from max_idx to max_idx + half_radius
                left3 = max(0, max_idx - peak_neighbor_radius)
                right3 = min(len(abs_signal), max_idx + peak_neighbor_radius + 1)
                left_signal = abs_signal[left3:max_idx]
                right_signal = abs_signal[max_idx + 1:right3]
                # Must be strictly increasing to avoid getting too close to nearby peaks
                is_left_increasing = np.all(np.diff(left_signal) > 0)
                is_right_decreasing = np.all(np.diff(right_signal) < 0)

                if max_val > threshold and is_left_increasing and is_right_decreasing and max_val > side_max * 2:
                    # Record the peak index
                    peaks.append(max_idx)
                else:
                    break

                # Suppress the neighborhood of the detected peak
            start = max(0, max_idx - process_radius)
            end = min(len(abs_signal), max_idx + process_radius + 1)
            abs_signal[start:end] = 0

        return sorted(peaks)

    @staticmethod
    def suppress_peaks(signal: np.ndarray, peak_indices: List[int],
                       neighbor_radius: int = 5, param: float = 1.2) -> np.ndarray:
        """
        Suppress specified peaks and their neighborhoods in the signal.
        :param signal: Input signal array.
        :param peak_indices: Indices of peaks to be suppressed.
        :param neighbor_radius: Minimum suppression radius.
        :param param: Scaling factor for the suppression radius.
        :return: A copy of the signal with suppressed regions zeroed out.
        """
        process_radius = int(neighbor_radius * param)
        signal_copy = signal.copy()
        for idx in peak_indices:
            start = max(0, idx - process_radius)
            end = min(len(signal_copy), idx + process_radius + 1)
            signal_copy[start:end] = 0
        return signal_copy

    @staticmethod
    def detect_breakpoints_manual(angles: np.ndarray,
                                  curvatures: np.ndarray,
                                  sigma: float = 5.0,
                                  neighbor_radius: int = 5,
                                  expected_corners: int = -1,
                                  expected_tangents: int = -1,
                                  param: float = 1.0,
                                  ignore_edge: int = 10,
                                  corner_thresh: Optional[float] = None,
                                  tangent_thresh: Optional[float] = None,
                                  plot_result: bool = False) -> SegmentInfo:
        """
        Detect corner and tangent points (breakpoints)
        :param angles: Array of tangent angles in radians.
        :param curvatures: Array of curvature values.
        :param sigma: Standard deviation for Gaussian smoothing.
        :param neighbor_radius: Neighborhood radius for analysis.
        :param expected_corners: Target number of corner points.
        :param expected_tangents: Target number of tangent points.
        :param param: Scaling factor for the suppression radius.
        :param ignore_edge: Number of points to ignore at the boundaries to avoid edge effects.
        :param corner_thresh: [Deprecated] Absolute threshold below which corners are ignored.
        :param tangent_thresh: [Deprecated] Absolute threshold below which tangents are ignored.
        :param plot_result: Whether to visualize the steps.
        :return: A SegmentInfo object containing detected indices.
        """
        # # Test code
        # if plot_result:
        #     curvs = curvatures
        #     diff1 = FeatureAnalysis.compute_value_diffs(angles, jump=1)
        #     diff2 = FeatureAnalysis.compute_value_diffs(diff1, jump=1)
        #     Plotter.plot_curve_with_breakpoints(curvs, segment_info=None, title="Curvatures")
        #     Plotter.plot_curve_with_breakpoints(diff1, segment_info=None, title="Angle first-order difference")
        #     Plotter.plot_curve_with_breakpoints(diff2, segment_info=None, title="Angle second-order difference")
        #     exit(0)

        # Compute all the signals
        diff1 = FeatureAnalysis.compute_value_diffs(angles, jump=1)
        diff1 = gaussian_filter1d(diff1, sigma=sigma, radius=neighbor_radius)

        diff2 = FeatureAnalysis.compute_value_diffs(diff1, jump=1)
        diff2 = gaussian_filter1d(diff2, sigma=sigma, radius=neighbor_radius)

        curvs = gaussian_filter1d(curvatures, sigma=sigma, radius=neighbor_radius)

        # if plot_result:
        #     Plotter.plot_curve_with_breakpoints(curvs, segment_info=None, title="Curvatures")
        #     Plotter.plot_curve_with_breakpoints(diff1, segment_info=None, title="Angle first-order difference")
        #     Plotter.plot_curve_with_breakpoints(diff2, segment_info=None, title="Angle second-order difference")

        if expected_corners != 0:
            # Detect corners
            corners = PointRecognizer.manual_peak_detection(signal=curvs, num_peaks=expected_corners, neighbor_radius=2 * neighbor_radius - 1, param=param, ignore_edge=ignore_edge, min_thresh=corner_thresh, plot_result=plot_result)
        else:
            corners = []

        if expected_tangents != 0:
            # Suppress regions around detected corners
            diff2_suppressed = PointRecognizer.suppress_peaks(diff2, corners, 3 * neighbor_radius + 1, param=param)

            # Detect tangent points
            tangents = PointRecognizer.manual_peak_detection(signal=diff2_suppressed, num_peaks=expected_tangents, neighbor_radius=3 * neighbor_radius + 1, param=param, ignore_edge=ignore_edge, min_thresh=tangent_thresh, plot_result=plot_result)
        else:
            tangents = []

        return SegmentInfo(corners, tangents)
