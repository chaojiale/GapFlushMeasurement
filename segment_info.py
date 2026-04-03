from typing import List, Optional


class SegmentInfo:
    def __init__(self, corners: Optional[List[int]] = None, tangents: Optional[List[int]] = None, line_types: Optional[str] = None, outer_pos: Optional[str] = None):
        """
        Initialize the segmentation information for a profile.
        :param corners: List of indices for corner points.
        :param tangents: List of indices for tangent points.
        :param line_types: String identifier for the geometric types of segments.
        :param outer_pos: Indicator for the outer surface position ('s' for start, 'e' for end).
        """
        self.corners = corners
        self.tangents = tangents
        self.line_types = line_types
        self.outer_pos = outer_pos

    def __str__(self) -> str:
        """
        Return a string representation of the segment information.
        """
        return f"corners: {self.corners}, tangents: {self.tangents}, line_types: {self.line_types}"

    def csv(self) -> str:
        """
        Format corner and tangent indices as a string in the result csv file.
        :return: Formatted string for data export.
        """
        corners_str = "[%s]" % ", ".join(f"{float(c):.2f}" for c in self.corners)
        tangents_str = "[%s]" % ", ".join(f"{float(t):.2f}" for t in self.tangents)
        return f"{corners_str}, {tangents_str}"
