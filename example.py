import numpy as np
from enums import FilterType, MeasuringMethod
from breakpoint_detection import BreakpointDetection
from flush_gap_tool import FlushGapTool
from section_info import SectionInfo
from section_model import SectionModel


if __name__ == '__main__':
    # Initialize section information.
    # If a section file is available, use: SectionInfo(section_name, section_path)

    # Generate synthetic door and fender profiles for demonstration.
    np.random.seed(0)
    door_points, _ = SectionModel.get_model("door", 0.08, 10)
    fender_points, _ = SectionModel.get_model("fender", 0.08, 10)
    door_points_3d = np.column_stack((door_points, np.zeros(len(door_points))))
    fender_points_3d = np.column_stack((fender_points, np.zeros(len(fender_points))))

    section_name = "example"
    section_info = SectionInfo(section_name, "")
    section_info.part1_points = door_points_3d
    section_info.part2_points = fender_points_3d
    section_info.initialize_rotated_point_cloud()

    direction = section_info.get_transformed_vector(np.array([0, 1, 0]))
    points1 = section_info.rotated_part1_points[:, :2]
    points2 = section_info.rotated_part2_points[:, :2]

    detection1 = BreakpointDetection(points1, 0, 2, "lal")
    _, segment1 = detection1.Execute(FilterType.ADAPTIVE_GAUSSIAN, 5, 5, 1.0, 5, None, False)
    segment1.outer_pos = 'e'
    detection2 = BreakpointDetection(points2, 0, 2, "lal")
    _, segment2 = detection2.Execute(FilterType.ADAPTIVE_GAUSSIAN, 5, 5, 1.0, 5, None, False)
    segment2.outer_pos = 's'

    acc = 3
    print("plane normal:", section_info.plane_normal_str())
    tool = FlushGapTool(points1, points2, segment1, segment2, section_name)
    flush1, gap1 = tool.calculate_flush_gap(MeasuringMethod.PRIMARY1, plot_result=True)
    print(f"Part1-primary flush: {flush1:.{acc}f}, gap: {gap1:.{acc}f}")
    flush2, gap2 = tool.calculate_flush_gap(MeasuringMethod.PRIMARY2, plot_result=True)
    print(f"Part2-primary flush: {-flush2:.{acc}f}, gap: {gap2:.{acc}f}")
    flush3, gap3 = tool.calculate_flush_gap(MeasuringMethod.BISECTOR, plot_result=True)
    print(f"Bisector flush: {flush3:.{acc}f}, gap: {gap3:.{acc}f}")
    flush4, gap4 = tool.calculate_flush_gap(MeasuringMethod.GLOBAL, direction, plot_result=True)
    print(f"Global {direction} flush: {flush4:.{acc}f}, gap: {gap4:.{acc}f}")
