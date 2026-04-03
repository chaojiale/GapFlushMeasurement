import numpy as np
import matplotlib.pyplot as plt

from enums import MeasuringMethod
from segment_info import SegmentInfo
from single_side import SingleSide
from geometry_tools import GeometryTools
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Arc, FancyArrowPatch
from typing import Optional, Tuple, Union

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 24

point_color = 'blue'
model_point_size = 2
signal_point_size = 4
breakpoint_color = 'red'
breakpoint_size = 50
acc = 3
axis_fontsize = 28
gf_line_width = 2.5
gf_point_size = 3


class Plotter:
    @staticmethod
    def plot_two_side_arrow(p1, p2, color='brown', linewidth=gf_line_width, arrow_size=15, min_length=1):
        p1 = np.array(p1)
        p2 = np.array(p2)
        ax = plt.gca()

        length = np.linalg.norm(p2 - p1)
        direction = (p2 - p1) / length if length != 0 else np.array([1, 0])

        if length < min_length:
            arrow_length = min_length * 0.005

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth, zorder=15)
            arrow1 = FancyArrowPatch(p1 - direction * arrow_length, p1,
                                     arrowstyle='->', color=color, linewidth=linewidth,
                                     mutation_scale=arrow_size, zorder=15)
            arrow2 = FancyArrowPatch(p2 + direction * arrow_length, p2,
                                     arrowstyle='->', color=color, linewidth=linewidth,
                                     mutation_scale=arrow_size, zorder=15)
            ax.add_patch(arrow1)
            ax.add_patch(arrow2)
        else:
            extension = arrow_size * 0.01
            new_p1 = p1 - direction * extension
            new_p2 = p2 + direction * extension

            arrow = FancyArrowPatch(new_p1, new_p2,
                                    arrowstyle='<->', color=color,
                                    linewidth=linewidth,
                                    mutation_scale=arrow_size, zorder=15)
            ax.add_patch(arrow)

    @staticmethod
    def plot_scatter_points(points, segment_info: Optional[SegmentInfo], title):
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], color=point_color, s=model_point_size, label='Points', zorder=3)

        if segment_info is not None:
            if segment_info.corners is not None:
                plt.scatter(points[segment_info.corners, 0], points[segment_info.corners, 1], color=breakpoint_color, s=breakpoint_size * 3, label='Corners', zorder=5, marker='x')
                # for i, idx in enumerate(segment_info.corners):
                #     plt.text(points[idx, 0], points[idx, 1], f'C{i}', color='black')

            if segment_info.tangents is not None:
                plt.scatter(points[segment_info.tangents, 0], points[segment_info.tangents, 1], color=breakpoint_color, s=breakpoint_size * 1.5, label='Tangents', zorder=5)
                # for i, idx in enumerate(segment_info.tangents):
                #     plt.text(points[idx, 0], points[idx, 1], f'T{i}', color='black')

        plt.title(title)
        plt.axis('equal')
        plt.show()

    @staticmethod
    def plot_flush_gap_preparing(part1: SingleSide, part2: SingleSide, section_name=None, wait_for=False):
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax = plt.gca()
        ax.axis('off')

        def plot_points(points, tangents, external, param, label, color=point_color, key_color=breakpoint_color):
            plt.scatter(points[:, 0], points[:, 1], color=color, s=gf_point_size, label=label, zorder=3)
            if tangents is not None:
                plt.scatter(points[tangents, 0], points[tangents, 1], color=key_color, s=breakpoint_size * 1.5, label='Tangents', zorder=8)
                # for i, idx in enumerate(tangents):
                #     plt.text(points[idx, 0], points[idx, 1], f'T{i}', color='black')

            if external is not None:
                # plt.scatter(external[:, 0], external[:, 1], color='green', s=model_point_size, label='External Points', zorder=4)

                if param is not None:
                    k, b = param
                    x_vals = np.linspace(min(external[:, 0]), max(external[:, 0]), 100)
                    y_vals = k * x_vals + b
                    plt.plot(x_vals, y_vals, color='black', label='Fitted Line', zorder=6, linewidth=gf_line_width)

        plot_points(part1.points, part1.segment.tangents, part1.external_points, (part1.k, part1.b), label='Primary Points', color='blue')
        plot_points(part2.points, part2.segment.tangents, part2.external_points, (part2.k, part2.b), label='Secondary Points', color='green')

        # title = f'Flush Gap Analysis - {section_name}' if section_name else 'Flush Gap Analysis'
        # plt.title(title)
        plt.axis('equal')
        # plt.grid(True, linestyle='--', alpha=0.6)
        if not wait_for:
            plt.show()

    @staticmethod
    def plot_flush(direction, p1, p2, param: Union[float, Tuple[float, float], np.ndarray], flush):
        if direction == MeasuringMethod.PRIMARY1:
            Plotter._plot_flush_master(p1, p2, param, flush)
        elif direction == MeasuringMethod.PRIMARY2:
            Plotter._plot_flush_master(p2, p1, param, flush)
        elif direction == MeasuringMethod.BISECTOR:
            Plotter._plot_flush_sharing(p1, p2, param, flush)
        elif direction == MeasuringMethod.GLOBAL:
            Plotter._plot_flush_absolute_direction(p1, p2, param, flush)
        else:
            raise ValueError("Invalid measuring direction")

    @staticmethod
    def _plot_flush_master(p1, p2, k1, flush):
        b1 = p1[1] - k1 * p1[0]
        proj = GeometryTools.get_projection_point(p2, k1, b1)
        extension_point = proj + (proj - p1) / 4
        plt.plot([p1[0], extension_point[0]], [p1[1], extension_point[1]], color='orange', linestyle='--', label='Projection Line', zorder=3, linewidth=gf_line_width)

        # plt.plot([p2[0], proj[0]], [p2[1], proj[1]], color='brown', linestyle='-', label='Tangent Projection Line', zorder=3, linewidth=1)
        Plotter.plot_two_side_arrow(p2, proj)

        # plt.text((p2[0] + proj[0]) / 2, (p2[1] + proj[1]) / 2, f'Flush: {flush:.{acc}f}', color='black', ha='center', va='center')

    @staticmethod
    def _plot_flush_sharing(p1, p2, k, flush):
        center = (p1 + p2) / 2
        b = center[1] - k * center[0]
        proj1 = GeometryTools.get_projection_point(p1, k, b)
        proj2 = GeometryTools.get_projection_point(p2, k, b)

        e1 = proj1 + (proj1 - proj2) / 4
        e2 = proj2 + (proj2 - proj1) / 4
        plt.plot([e1[0], e2[0]], [e1[1], e2[1]], color='orange', linestyle='--', label='Projection Line', zorder=3, linewidth=gf_line_width)

        # plt.plot([p1[0], proj1[0]], [p1[1], proj1[1]], color='brown', linestyle='-', label='Tangent Projection Line 1', zorder=3, linewidth=1)
        Plotter.plot_two_side_arrow(p1, proj1)
        # d1 = np.abs(k * p1[0] - p1[1] + b) / np.sqrt(k ** 2 + 1)
        # plt.text((p1[0] + proj1[0]) / 2, (p1[1] + proj1[1]) / 2, f'Flush: {d1:.{acc}f}', color='black', ha='center', va='center')

        # plt.plot([p2[0], proj2[0]], [p2[1], proj2[1]], color='brown', linestyle='-', label='Tangent Projection Line 2', zorder=3, linewidth=1)
        Plotter.plot_two_side_arrow(p2, proj2)
        # d2 = np.abs(k * p2[0] - p2[1] + b) / np.sqrt(k ** 2 + 1)
        # plt.text((p2[0] + proj2[0]) / 2, (p2[1] + proj2[1]) / 2, f'Flush: {d2:.{acc}f}', color='black', ha='center', va='center')
        # plt.text(center[0], center[1], f'Flush: {flush:.{acc}f}', color='black', ha='center', va='center')

    @staticmethod
    def _plot_flush_absolute_direction(p1, p2, vec, flush):

        # plt.quiver(p2[0], p2[1], p1[0] - p2[0], p1[1] - p2[1],
        #            angles='xy', scale_units='xy', scale=1, color='brown', label='Tangent Vector', zorder=3, width=0.005)

        p_ref = (p1 + p2) / 2
        plt.quiver(p_ref[0], p_ref[1], vec[0] * 2, vec[1] * 2,
                   angles='xy', scale_units='xy', scale=1, color='black', label='Measure Vector', zorder=3)

        proj = GeometryTools.get_projection_point2(p1, p2, vec)
        # plt.plot([p2[0], proj[0]], [p2[1], proj[1]], color='brown', linestyle='-', label='Projection Line', linewidth=1)
        Plotter.plot_two_side_arrow(p2, proj)
        # plt.text((p2[0] + proj[0]) / 2, (p2[1] + proj[1]) / 2, f'Flush: {flush:.{acc}f}', color='black', ha='center', va='center')

        extent = proj + (proj - p1) / 4
        plt.plot([p1[0], extent[0]], [p1[1], extent[1]], color='orange', linestyle='--', linewidth=gf_line_width)

    @staticmethod
    def _plot_vertical_lines(length, vec, s1, s2):
        line_start1 = s1 + vec * length / 2
        line_end1 = s1 - vec * length / 2
        line_start2 = s2 + vec * length / 2
        line_end2 = s2 - vec * length / 2
        plt.plot([line_start1[0], line_end1[0]], [line_start1[1], line_end1[1]], color='blue', linestyle='--', label='Master Line', zorder=3, linewidth=gf_line_width)
        plt.plot([line_start2[0], line_end2[0]], [line_start2[1], line_end2[1]], color='green', linestyle='--', label='Slave Line', zorder=3, linewidth=gf_line_width)

    @staticmethod
    def _plot_gap_line(s1, s2, vec):
        proj_length = np.dot(s1 - s2, vec)
        pp1 = s1 - vec * proj_length / 2
        pp2 = s2 + vec * proj_length / 2
        # plt.plot([pp1[0], pp2[0]], [pp1[1], pp2[1]], color='brown', linestyle='-', label='Gap Line', zorder=3, linewidth=1)
        Plotter.plot_two_side_arrow(pp1, pp2)
        # arrow = FancyArrowPatch(pp1, pp2, arrowstyle='<->', color='brown', linewidth=1, zorder=3, mutation_scale=20)
        # plt.gca().add_patch(arrow)

    @staticmethod
    def plot_gap(direction, p1, p2, param: Union[Tuple[float, float], np.ndarray], s1, s2, gap):
        if direction == MeasuringMethod.PRIMARY1:
            Plotter._plot_gap_master(p1, p2, param, s1, s2, gap)
        elif direction == MeasuringMethod.PRIMARY2:
            Plotter._plot_gap_master(p2, p1, param, s2, s1, gap)
        elif direction == MeasuringMethod.BISECTOR:
            Plotter._plot_gap_sharing(p1, p2, param, s1, s2, gap)
        elif direction == MeasuringMethod.GLOBAL:
            Plotter._plot_gap_absolute_direction(p1, p2, param, s1, s2, gap)
        else:
            raise ValueError("Invalid measuring direction")

    @staticmethod
    def _plot_gap_master(p1, p2, param, s1, s2, gap):
        k1 = param[1]
        b1 = p1[1] - k1 * p1[0]
        proj = GeometryTools.get_projection_point(p2, k1, b1)
        extension_point = proj + (proj - p1) / 4
        # plt.plot([p1[0], extension_point[0]], [p1[1], extension_point[1]], color='orange', linestyle=':', label='Projection Line', zorder=3)
        length = np.linalg.norm(p1 - p2)
        vec = np.array([-k1, 1])
        vec /= np.linalg.norm(vec)
        Plotter._plot_vertical_lines(length, vec, s1, s2)
        Plotter._plot_gap_line(s1, s2, vec)

        # s = np.array([s1, s2])
        # plt.scatter(s[:, 0], s[:, 1], color='red', s=breakpoint_size * 1.5, label='Tangents', zorder=8)
        # plt.scatter(s[:, 0], s[:, 1], color='red', s=breakpoint_size * 1.5, label='Tangents', zorder=8)

        # p_ref = (p1 + p2) / 2
        # plt.text(p_ref[0], p_ref[1], f'Gap: {gap:.{acc}f}', color='black', ha='center', va='center')

    @staticmethod
    def _plot_gap_sharing(p1, p2, param, s1, s2, gap):
        center = (p1 + p2) / 2
        k = param[1]
        b = center[1] - k * center[0]
        proj1 = GeometryTools.get_projection_point(p1, k, b)
        proj2 = GeometryTools.get_projection_point(p2, k, b)

        e1 = proj1 + (proj1 - proj2) / 4
        e2 = proj2 + (proj2 - proj1) / 4
        # plt.plot([e1[0], e2[0]], [e1[1], e2[1]], color='orange', linestyle=':', label='Projection Line', zorder=3)

        length = np.linalg.norm(p1 - p2)
        vec = np.array([-k, 1])
        vec /= np.linalg.norm(vec)
        Plotter._plot_vertical_lines(length, vec, s1, s2)
        Plotter._plot_gap_line(s1, s2, vec)

        # p_ref = (p1 + p2) / 2
        # plt.text(p_ref[0], p_ref[1], f'Gap: {gap:.{acc}f}', color='black', ha='center', va='center')

    @staticmethod
    def _plot_gap_absolute_direction(p1, p2, param, s1, s2, gap):
        length = np.linalg.norm(p1 - p2)
        param /= np.linalg.norm(param)
        vec = np.array([param[1], -param[0]])

        Plotter._plot_vertical_lines(length, vec, s1, s2)
        Plotter._plot_gap_line(s1, s2, vec)

        p_ref = (p1 + p2) / 2
        draw_start = p_ref - param * length / 4
        # plt.quiver(draw_start[0], draw_start[1], param[0] * length / 2, param[1] * length / 2, angles='xy', scale_units='xy', scale=1, color='red', label='Tangent Vector', zorder=3)

        # plt.text(p_ref[0], p_ref[1], f'Gap: {gap:.{acc}f}', color='black', ha='center', va='center')

    @staticmethod
    def plot_gap_flush_submit():
        plt.show()

    @staticmethod
    def plot_curve_with_breakpoints(values, segment_info: Optional[SegmentInfo], title, label="Value"):
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(values)), values, color=point_color, s=signal_point_size, label=label, zorder=1)

        if segment_info is not None:
            if segment_info.corners is not None:
                plt.scatter(segment_info.corners, [values[i] for i in segment_info.corners], color=breakpoint_color, s=breakpoint_size, label='Corners', zorder=3)
                for i, idx in enumerate(segment_info.corners):
                    plt.text(idx, values[idx], f'C{i}', color='black', ha='center', va='bottom')

            if segment_info.tangents is not None:
                plt.scatter(segment_info.tangents, [values[i] for i in segment_info.tangents], color=breakpoint_color, s=breakpoint_size, label='Tangents', zorder=3)
                for i, idx in enumerate(segment_info.tangents):
                    plt.text(idx, values[idx], f'T{i}', color='black', ha='center', va='bottom')

        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        # ax.yaxis.set_major_locator(AutoLocator())

        plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)
        plt.xlabel('Index', fontsize=axis_fontsize)
        plt.ylabel(label, fontsize=axis_fontsize)
        # plt.title(title)
        # plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    @staticmethod
    def plot_curve_with_two_breakpoints(values, ref_key_point_info, find_key_point_info, title, label="Value"):
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(values)), values, color=point_color, s=signal_point_size, label=label, zorder=1)

        if ref_key_point_info is not None:
            plt.scatter(ref_key_point_info.corners, [values[i] for i in ref_key_point_info.corners], color=breakpoint_color, s=breakpoint_size, label='Corners', zorder=3)
            for i, idx in enumerate(ref_key_point_info.corners):
                plt.text(idx, values[idx], f'C{i}', color='black', ha='center', va='bottom')

            plt.scatter(ref_key_point_info.tangents, [values[i] for i in ref_key_point_info.tangents], color=breakpoint_color, s=breakpoint_size, label='Tangents', zorder=3)
            for i, idx in enumerate(ref_key_point_info.tangents):
                plt.text(idx, values[idx], f'T{i}', color='black', ha='center', va='bottom')

        if find_key_point_info is not None:
            plt.scatter(find_key_point_info.corners, [values[i] for i in find_key_point_info.corners], color='green', s=breakpoint_size, label='Corners', zorder=3)
            for i, idx in enumerate(find_key_point_info.corners):
                plt.text(idx, values[idx], f'C{i}', color='black', ha='center', va='bottom')

            plt.scatter(find_key_point_info.tangents, [values[i] for i in find_key_point_info.tangents], color='green', s=breakpoint_size, label='Tangents', zorder=3)
            for i, idx in enumerate(find_key_point_info.tangents):
                plt.text(idx, values[idx], f'T{i}', color='black', ha='center', va='bottom')

        plt.xlabel('Index')
        plt.ylabel(label)
        plt.title(title)
        # plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    @staticmethod
    def plot_scatter_points_with_one_normal(points, point_idx, half_size, normal, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], color=point_color, s=model_point_size, label='Points', zorder=3)
        plt.scatter(points[point_idx - half_size:point_idx + half_size + 1, 0], points[point_idx - half_size:point_idx + half_size + 1, 1], color='orange', s=model_point_size + 1, label='Neighbor Points', zorder=3)
        plt.scatter(points[point_idx, 0], points[point_idx, 1], color=breakpoint_color, s=breakpoint_size / 2, label='Key Point', zorder=5)

        normal = normal / np.linalg.norm(normal) * 20
        plt.quiver(points[point_idx, 0] + 0.5, points[point_idx, 1], normal[0], normal[1],
                   angles='xy', scale_units='xy', scale=1, color='green', label='Normal Vector', width=0.005, zorder=4)
        tangent = np.array([-normal[1], normal[0]])
        line_s = points[point_idx] + tangent
        line_e = points[point_idx] - tangent
        plt.plot([line_s[0], line_e[0]], [line_s[1], line_e[1]], color='green', linestyle='--', label='Normal Vector', zorder=4)

        h_line_start = line_s
        h_line_end = h_line_start + np.array([25, 0])
        plt.plot([h_line_start[0], h_line_end[0]], [h_line_start[1], h_line_end[1]], color='black', linestyle='-', label='Horizontal Line', zorder=4, linewidth=0.4)
        extend_end = h_line_start + tangent * 1.25
        plt.plot([h_line_start[0], extend_end[0]], [h_line_start[1], extend_end[1]], color='black', linestyle='-', label='Extended Line', zorder=4, linewidth=0.4)

        plt.title(title)
        plt.axis('equal')
        plt.show()

    @staticmethod
    def plot_scatter_points_with_curvatures(points, point_idx, half_size, normals, curvatures, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], color=point_color, s=model_point_size, label='Points', zorder=3)

        line_ends = []
        for i in range(point_idx - half_size, point_idx + half_size + 1):
            current_normal = np.array([normals[i][0], normals[i][1]])
            curv = curvatures[i] * 800
            point = points[i, :]
            line_end = point + current_normal * curv
            line_ends.append(line_end)
            plt.plot([point[0], line_end[0]], [point[1], line_end[1]], color='green', linestyle='-', label='Normal Vector', zorder=4, linewidth=0.8)

        for i in range(len(line_ends) - 1):
            plt.plot([line_ends[i][0], line_ends[i + 1][0]], [line_ends[i][1], line_ends[i + 1][1]], color='black', linestyle=':', zorder=4, linewidth=1)

        plt.title(title)
        plt.axis('equal')
        plt.show()

    @staticmethod
    def plot_points_with_shapes(points, mid, half_size, left, right, line_param1, arc_param1, line_param2, arc_param2, title):
        plt.figure(figsize=(8, 6))

        plt.scatter(points[:, 0], points[:, 1], color=point_color, s=model_point_size, label='Points', zorder=3)
        plt.scatter(points[mid - half_size:mid + half_size + 1, 0], points[mid - half_size:mid + half_size + 1, 1], color='orange', s=model_point_size + 5, label='Neighbor Points', zorder=3)

        key_points = points[[left, mid, right], :]
        plt.scatter(key_points[:, 0], key_points[:, 1], color=breakpoint_color, s=breakpoint_size / 2, label='Key Point', zorder=7)
        s1, e1 = line_param1
        plt.plot([s1[0], e1[0]], [s1[1], e1[1]], color=(1.0, 0, 1.0), linestyle='-', label='Line 1', zorder=6, linewidth=0.5)
        c1, r1, sa1, se1 = arc_param1
        arc1 = Arc(
            (c1[0], c1[1]),
            2 * r1, 2 * r1,
            theta1=se1 * 180 / np.pi,
            theta2=sa1 * 180 / np.pi,
            color='green',
            linewidth=2,
            zorder=5,
        )
        plt.gca().add_patch(arc1)

        s2, e2 = line_param2
        plt.plot([s2[0], e2[0]], [s2[1], e2[1]], color='green', linestyle='-', label='Line 2', zorder=5, linewidth=2)
        c2, r2, sa2, se2 = arc_param2
        arc2 = Arc(
            (c2[0], c2[1]),
            2 * r2, 2 * r2,
            theta1=sa2 * 180 / np.pi,
            theta2=se2 * 180 / np.pi,
            color=(1.0, 0, 1.0),
            linewidth=0.5,
            zorder=6,
        )
        plt.gca().add_patch(arc2)

        # Plotter.plot_points_with_shapes(filtered_points, mid, half_size, left, right, [s1, e1], [c1, r1, sa1, se1], [s2, e2], [c2, r2, sa2, se2], title="Optimization Process")
        plt.title(title)
        plt.axis('equal')
        plt.show()
