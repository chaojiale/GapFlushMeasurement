from enum import Enum


class FilterType(Enum):
    DISABLE = "Disable"
    MEAN = "Mean"
    GAUSSIAN = "Gaussian"
    ADAPTIVE_GAUSSIAN = "Adaptive gaussian"
    BILATERAL = "Bilateral"
    GUIDED = "Guided"

    def __str__(self):
        return self.value


class FeatureType(Enum):
    VARIATION = "variation"
    DIRECT_ANGLE = "direct angle"
    DIRECT_ANGLE_DIFF = "direct angle difference"
    ANGLE = "fitting angle"
    ANGLE_DIFF = "fitting angle difference"

    def __str__(self):
        return self.value


class MeasuringMethod(Enum):
    PRIMARY1 = "Part1 primary"
    PRIMARY2 = "Part2 primary"
    BISECTOR = "Bisector"
    GLOBAL = "Global"
