# src/strategies/greedy_placement.py
import numpy as np
from src.config.config import BRICK_SIZES, LEGO_COLORS
from src.strategies.base import PlacementStrategy
from src.strategies.utils import can_place_brick, place_bricks_on_layer_fast
from numba.typed import List as NumbaList

class GreedyPlacementStrategy(PlacementStrategy):
    def __init__(self):
        # Сортируем размеры кирпичей по убыванию объема один раз при инициализации
        self.sorted_sizes = sorted(BRICK_SIZES, key=lambda s: s[0] * s[1] * s[2], reverse=True)
        self.numba_sorted_sizes = NumbaList([(w, h, d, t) for w, h, d, t in self.sorted_sizes])

    def place_bricks(self, voxel_array, use_colors, allowed_sizes=None, allow_top_layer=False, 
                     progress_callback=None, brick_type=None):
        voxel_copy = voxel_array.copy()
        support_array = np.zeros_like(voxel_array, dtype=bool)
        cubes = []
        total_voxels = np.sum(voxel_array)
        processed_voxels = 0
        
        # Используем предсортированные размеры, если allowed_sizes не задан
        allowed_sizes = allowed_sizes or self.sorted_sizes
        numba_allowed_sizes = NumbaList([(w, h, d, t) for w, h, d, t in allowed_sizes])

        # Минимальный кирпич для ранней остановки (предполагаем, что 1x1x1 есть в allowed_sizes)
        min_brick = (1, 1, 1, "brick")

        for z in range(voxel_array.shape[0]):
            if progress_callback and progress_callback(processed_voxels / total_voxels):
                return cubes
            
            # Ранняя остановка: проверяем минимальный кирпич перед полным проходом
            layer_voxels = voxel_copy[z]
            if np.any(layer_voxels):
                # Проверяем, можно ли разместить минимальный кирпич где-то на слое
                can_place_min = False
                for y in range(layer_voxels.shape[0]):
                    for x in range(layer_voxels.shape[1]):
                        if layer_voxels[y, x] and can_place_brick(x, y, z, *min_brick[:3], 
                                                                 voxel_copy, support_array, allow_top_layer):
                            can_place_min = True
                            break
                    if can_place_min:
                        break
                if not can_place_min:
                    continue  # Пропускаем слой, если даже минимальный кирпич не помещается

            layer_cubes = place_bricks_on_layer_fast(z, voxel_copy, support_array, numba_allowed_sizes, allow_top_layer)
            for cube in layer_cubes:
                x, y, z_local, w, h, d, placed_brick_type = cube
                color = np.random.choice(LEGO_COLORS) if use_colors else "#000000"
                final_brick_type = brick_type if brick_type is not None else placed_brick_type
                cubes.append((x, y, z, w, h, d, color, final_brick_type))
                processed_voxels += w * h * d
        
        return cubes
