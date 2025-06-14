# src/strategies/branch_and_bound_placement.py
import numpy as np
import logging
from typing import List, Tuple
from heapq import heappush, heappop
from numba import njit
from src.config.config import BRICK_PROPERTIES, LEGO_COLORS
from src.strategies.base import PlacementStrategy
from src.strategies.utils import can_place_brick, find_next_voxel, place_brick

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@njit
def compute_heuristic(remaining_voxels: int, cubes: List[Tuple[int, int, int, int, int, int, str]], 
                      voxel_shape: Tuple[int, int, int]) -> float:
    """Numba-версия эвристики для ускорения."""
    if remaining_voxels == 0:
        return 0
    
    max_brick_volume = 24  # Максимальный объем из BRICK_PROPERTIES (2x4x3, например)
    volume_cost = remaining_voxels / max_brick_volume

    # Оценка числа кирпичей (упрощена для Numba, без связности)
    brick_count_penalty = len(cubes) * 0.1
    
    return volume_cost + brick_count_penalty

class BranchAndBoundPlacementStrategy(PlacementStrategy):
    def place_bricks(self, voxel_array, use_colors, allowed_sizes, allow_top_layer=False, 
                     progress_callback=None, brick_type=None):
        voxel_copy = voxel_array  # In-place, копия не создается
        support_array = np.zeros_like(voxel_array, dtype=bool)
        total_voxels = np.sum(voxel_array)
        processed_voxels = 0
        allowed_sizes = sorted(allowed_sizes, key=lambda s: s[0] * s[1] * s[2], reverse=True)
        best_cubes = []
        visited_voxels = set()

        # Жадная инициализация
        initial_cubes = []
        for z in range(voxel_copy.shape[0]):
            while np.any(voxel_copy[z]):
                x, y, z_found, found = find_next_voxel_in_layer(voxel_copy, z)
                if not found:
                    break
                voxel_key = (x, y, z)
                if voxel_key in visited_voxels:
                    voxel_copy[z, y, x] = 0
                    continue
                visited_voxels.add(voxel_key)
                placed = False
                for w, h, d, t in allowed_sizes:
                    if can_place_brick(x, y, z, w, h, d, voxel_copy, support_array, allow_top_layer):
                        place_brick(x, y, z, w, h, d, voxel_copy, support_array)
                        initial_cubes.append((x, y, z, w, h, d, t))
                        processed_voxels += w * h * d
                        placed = True
                        break
                if not placed:
                    voxel_copy[z, y, x] = 0

        initial_state = (voxel_copy, support_array, initial_cubes)
        state_key = self._state_key(initial_state)
        h = compute_heuristic(np.sum(voxel_copy), initial_cubes, voxel_copy.shape)
        open_set = [(0, 0, initial_state)]
        g_score = {state_key: 0}
        f_score = {state_key: h}
        best_cubes = initial_cubes

        max_iterations = 10000
        iteration = 0
        while open_set and iteration < max_iterations:
            iteration += 1
            if progress_callback and progress_callback(processed_voxels / total_voxels):
                return self._finalize_cubes(best_cubes, use_colors, brick_type)
            f, g, (current_voxel, current_support, cubes) = heappop(open_set)
            remaining_voxels = np.sum(current_voxel)

            if remaining_voxels == 0:
                return self._finalize_cubes(cubes, use_colors, brick_type)

            x, y, z, found = find_next_voxel(current_voxel)
            if not found:
                if len(cubes) > len(best_cubes):
                    best_cubes = cubes
                continue

            max_depth = max(1, int(remaining_voxels / (total_voxels * 0.1)))
            if g > max_depth:
                continue

            for w, h, d, t in allowed_sizes:
                if can_place_brick(x, y, z, w, h, d, current_voxel, current_support, allow_top_layer):
                    # In-place модификация для экономии памяти
                    new_voxel = current_voxel.copy()  # Копируем только здесь
                    new_support = current_support.copy()
                    place_brick(x, y, z, w, h, d, new_voxel, new_support)
                    new_cubes = cubes + [(x, y, z, w, h, d, t)]
                    new_state = (new_voxel, new_support, new_cubes)

                    new_g = g + 1
                    new_h = compute_heuristic(np.sum(new_voxel), new_cubes, new_voxel.shape)
                    new_f = new_g + new_h

                    state_key = self._state_key(new_state)
                    if state_key not in g_score or new_g < g_score[state_key]:
                        g_score[state_key] = new_g
                        f_score[state_key] = new_f
                        heappush(open_set, (new_f, new_g, new_state))
                        processed_voxels = total_voxels - remaining_voxels
                        if len(new_cubes) > len(best_cubes):
                            best_cubes = new_cubes

        return self._finalize_cubes(best_cubes, use_colors, brick_type)

    def _state_key(self, state):
        _, _, cubes = state
        return tuple((x, y, z, w, h, d, t) for x, y, z, w, h, d, t in cubes)

    def _finalize_cubes(self, cubes, use_colors, brick_type):
        return [(x, y, z, w, h, d, np.random.choice(LEGO_COLORS) if use_colors else "#000000", brick_type or t)
                for x, y, z, w, h, d, t in cubes]

def find_next_voxel_in_layer(voxel_array: np.ndarray, z: int) -> Tuple[int, int, int, bool]:
    height, width = voxel_array.shape[1], voxel_array.shape[2]
    for y in range(height):
        for x in range(width):
            if voxel_array[z, y, x]:
                return x, y, z, True
    return -1, -1, z, False