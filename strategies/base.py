from abc import ABC, abstractmethod

class PlacementStrategy(ABC):
    @abstractmethod
    def place_bricks(self, voxel_array, use_colors, allowed_sizes, allow_top_layer=False, progress_callback=None, brick_type=None):
        pass