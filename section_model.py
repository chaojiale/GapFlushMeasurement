import numpy as np
from segment_info import SegmentInfo


class SectionModel:
    @staticmethod
    def generate_arc_number(center, radius, start_angle, end_angle, num_points, end_point=False) -> np.ndarray:
        angles = np.linspace(start_angle, end_angle, num_points, end_point)
        points = np.array([[center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)] for a in angles])
        return points

    @staticmethod
    def generate_arc_spacing(center, radius, start_angle, end_angle, spacing, end_point=False) -> np.ndarray:
        ideal_number = np.abs(end_angle - start_angle) / spacing * radius
        real_number = np.round(ideal_number).astype(int)
        return SectionModel.generate_arc_number(center, radius, start_angle, end_angle, real_number, end_point)

    @staticmethod
    def generate_line_number(start, end, num_points, end_point=False) -> np.ndarray:
        points = np.linspace(start, end, num_points, end_point)
        return points

    @staticmethod
    def generate_line_spacing(start, end, spacing, end_point=False) -> np.ndarray:
        ideal_number = np.linalg.norm(end - start) / spacing
        real_number = np.round(ideal_number).astype(int)
        return SectionModel.generate_line_number(start, end, real_number, end_point)

    @staticmethod
    def fender_model(used_spacing):
        arc_radius = 1
        line1_end = np.array([-2 + arc_radius * np.sin(1 / 36.0), -1 + arc_radius * np.cos(1 / 36.0)])
        line1_start = line1_end - 5 * np.array([np.cos(1 / 36.0), -np.sin(1 / 36.0)])
        line1 = SectionModel.generate_line_spacing(start=line1_start, end=line1_end, spacing=used_spacing, end_point=True)

        arc_center = np.array([-2, -1])
        arc_start_angle = 17 * np.pi / 36
        arc_end_angle = -np.pi / 6
        arc = SectionModel.generate_arc_spacing(center=arc_center, radius=arc_radius, start_angle=arc_start_angle, end_angle=arc_end_angle, spacing=used_spacing, end_point=False)

        line2_start = np.array([-2 + np.sqrt(3) / 2, -1.5])
        line2_end = line2_start + 4 * np.array([-1 / 2, -np.sqrt(3) / 2])
        line2 = SectionModel.generate_line_spacing(start=line2_start, end=line2_end, spacing=used_spacing, end_point=True)

        points = np.vstack([line1, arc, line2])
        tangents = [len(line1), len(line1) + len(arc)]
        return points, SegmentInfo([], tangents, "lal")

    @staticmethod
    def door_model(used_spacing):
        arc_radius = 0.7
        line1_start = np.array([6.5, -0.7])
        line1_end = np.array([2, -0.7])
        line1 = SectionModel.generate_line_spacing(start=line1_start, end=line1_end, spacing=used_spacing, end_point=True)

        arc_center = np.array([2, 0])
        arc_start_angle = -np.pi / 2
        arc_end_angle = -53 * np.pi / 36
        arc = SectionModel.generate_arc_spacing(center=arc_center, radius=arc_radius, start_angle=arc_start_angle, end_angle=arc_end_angle, spacing=used_spacing, end_point=False)

        line2_start = np.array([2 - arc_radius * np.sin(1 / 36.0), arc_radius * np.cos(1 / 36.0)])
        line2_end = line2_start + 5 * np.array([np.cos(1 / 36.0), np.sin(1 / 36.0)])
        line2 = SectionModel.generate_line_spacing(start=line2_start, end=line2_end, spacing=used_spacing, end_point=True)

        points = np.vstack([line1, arc, line2])
        tangents = [len(line1), len(line1) + len(arc)]
        return points, SegmentInfo([], tangents, "lal")

    @staticmethod
    def add_noise(points, snr):
        num_points = len(points)
        distances = np.zeros(num_points)
        distances[0] = np.linalg.norm(points[1] - points[0])
        distances[num_points - 1] = np.linalg.norm(points[-1] - points[-2])
        for i in range(1, num_points - 1):
            dist_prev = np.linalg.norm(points[i] - points[i - 1])
            dist_next = np.linalg.norm(points[i + 1] - points[i])
            distances[i] = (dist_prev + dist_next) / 2

        P_signal = (distances / 2) ** 2
        P_noise = P_signal / (10 ** (snr / 10))
        sigma_noise = np.sqrt(P_noise)
        noises = np.random.normal(0, sigma_noise[:, None], size=points.shape)
        return points + noises

    @staticmethod
    def get_model(model: str = "door", used_spacing=1.0, snr=None):
        if model == "fender":
            points, refer_info = SectionModel.fender_model(used_spacing)
        elif model == "door":
            points, refer_info = SectionModel.door_model(used_spacing)
        else:
            raise ValueError("Invalid model")

        if snr is not None:
            points = SectionModel.add_noise(points, snr)

        return points, refer_info
