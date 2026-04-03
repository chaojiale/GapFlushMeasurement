import numpy as np
from segment_info import SegmentInfo
from geometry_tools import GeometryTools


class SingleSide:
    def __init__(self, points: np.ndarray, segment: SegmentInfo):
        """
        Initialize the external surface information for a single side.
        :param points: Full point set of the profile.
        :param segment: Segmentation information.
        """
        self.points = points
        self.segment = segment
        # Determine the index of the tangent point based on the outer position ('s' for start, 'e' for end)
        self.tangent_idx = segment.tangents[0] if segment.outer_pos == 's' else segment.tangents[-1]
        # Extract points belonging to the external segment
        self.external_points = points[0:self.tangent_idx + 1] if segment.outer_pos == 's' else points[self.tangent_idx:]
        # Perform linear fitting on the extracted external points
        # Note: Consider using a general form equation (Ax + By + C = 0)
        # to prevent numerical instability when the slope k is extremely large.
        self.k, self.b = GeometryTools.fit_line(self.external_points)
        # Store the tangent point and its orthogonal projection
        self.tangent_point = points[self.tangent_idx]
        self.tangent_projection = GeometryTools.get_projection_point(self.tangent_point, self.k, self.b)
