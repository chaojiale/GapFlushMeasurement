
class Normal:
    def __init__(self, x: float = 0.0, y: float = 0.0, variation: float = 0.0):
        """
        Initialize the Normal object representing a 2D vector and its associated curve variation.
        :param x: The x-component of the normal vector.
        :param y: The y-component of the normal vector.
        :param variation: The curve variation.
        """
        self.x = x
        self.y = y
        self.variation = variation

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise IndexError("Invalid index")
