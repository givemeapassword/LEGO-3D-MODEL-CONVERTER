# brick_optimization.py
import numpy as np
from numba.typed import List as NumbaList
from typing import List, Tuple
import concurrent.futures
import logging
import gc
from scipy.sparse import coo_array  # Для разреженных структур
from src.config.config import BRICK_SIZES, LEGO_COLORS, STUD_SIZE, get_brick_height
from src.strategies.base import PlacementStrategy
from .strategies.greedy_placement import GreedyPlacementStrategy
from .strategies.simulated_annealing_placement import SimulatedAnnealingPlacementStrategy
from .strategies.branch_and_bound_placement import BranchAndBoundPlacementStrategy

MIN_BLOCK_SIZE = 5
MAX_BLOCK_SIZE = 20
DENSE_THRESHOLD = 0.5
SPARSE_THRESHOLD = 0.1

def get_block_size(voxel_shape: Tuple[int, int, int]) -> int:
    avg_dim = sum(voxel_shape) / 3
    return max(MIN_BLOCK_SIZE, min(MAX_BLOCK_SIZE, int(avg_dim / 10)))

def analyze_voxel_density(voxel_array: np.ndarray, base_size: int) -> List[Tuple[int, int, int, int, int, int]]:
    nz, ny, nx = voxel_array.shape
    blocks = []
    z, y, x = 0, 0, 0
    while z < nz:
        y = 0
        while y < ny:
            x = 0
            while x < nx:
                z_size = min(base_size, nz - z)
                y_size = min(base_size, ny - y)
                x_size = min(base_size, nx - x)
                blocks.append((z, y, x, z_size, y_size, x_size))
                x += x_size
            y += y_size
        z += z_size
    return blocks

def fill_hollow_model(voxel_array: np.ndarray, minimal_support: bool = False, inplace: bool = True) -> np.ndarray:
    if not inplace:
        filled_array = voxel_array.copy()
    else:
        filled_array = voxel_array
    nz, ny, nx = voxel_array.shape
    for y in range(ny):
        for x in range(nx):
            for z in range(1, nz):
                if filled_array[z, y, x]:
                    for z_below in range(z):
                        filled_array[z_below, y, x] = True
    return filled_array

def _process_block(args: Tuple[np.ndarray, bool, List[Tuple[int, int, int, str]], int, int, int, int, int, int, str, bool, bool]) -> Tuple[List[Tuple], int, int, int]:
    voxel_array, use_colors, allowed_sizes, z, y, x, z_size, y_size, x_size, strategy_name, fill_hollow, minimal_support = args
    block_id = f"z{z}_y{y}_x{x}"
    sub_voxel = voxel_array[z:z + z_size, y:y + y_size, x:x + x_size]
    
    logging.info(f"Processing block {block_id} - shape: {sub_voxel.shape}, voxels: {np.sum(sub_voxel)}")
    
    if not np.any(sub_voxel):
        logging.info(f"Block {block_id} empty, skipping")
        return [], z, y, x
    
    numba_allowed_sizes = NumbaList([(w, h, d, t) for w, h, d, t in allowed_sizes])
    strategies = {
        "greedy": GreedyPlacementStrategy,
        "simulated_annealing": SimulatedAnnealingPlacementStrategy,
        "branch_and_bound": BranchAndBoundPlacementStrategy
    }
    strategy = strategies.get(strategy_name, GreedyPlacementStrategy)()
    
    if fill_hollow:
        logging.info(f"Filling block {block_id}")
        sub_voxel = fill_hollow_model(sub_voxel, minimal_support=minimal_support, inplace=True)
    
    logging.info(f"Placing bricks in block {block_id}")
    local_cubes = strategy.place_bricks(sub_voxel, use_colors, allowed_sizes, not fill_hollow)
    
    logging.info(f"Block {block_id} processed: {len(local_cubes)} bricks")
    return local_cubes, z, y, x

class BrickPlacer:
    def __init__(self, strategy: PlacementStrategy):
        self.strategy = strategy
        self.allowed_sizes = None

    def _create_strategy(self, strategy_name: str) -> PlacementStrategy:
        strategies = {
            "greedy": GreedyPlacementStrategy,
            "simulated_annealing": SimulatedAnnealingPlacementStrategy,
            "branch_and_bound": BranchAndBoundPlacementStrategy
        }
        return strategies.get(strategy_name, GreedyPlacementStrategy)()

    def place_bricks(self, voxel_array: np.ndarray, use_colors: bool = True, allowed_sizes=None, 
                        fill_hollow: bool = True, minimal_support: bool = False, progress_callback=None, 
                        voxel_size: float = STUD_SIZE) -> List[Tuple]:
            self.allowed_sizes = [(w, h, d, t) for w, h, d, t in (allowed_sizes or BRICK_SIZES)]
            voxel_array = voxel_array.copy()
            nz, ny, nx = voxel_array.shape

            if fill_hollow:
                voxel_array = fill_hollow_model(voxel_array, minimal_support=False, inplace=True)
                logging.info("Model filled in-place: full fill")

            all_cubes = []
            occupied = np.zeros((nz, ny, nx), dtype=bool)

            # Проходим по слоям снизу вверх
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        if voxel_array[z, y, x] and not occupied[z, y, x]:
                            for w, h, d, brick_type in sorted(self.allowed_sizes, key=lambda x: x[0] * x[1], reverse=True):
                                brick_height_mm = get_brick_height(brick_type)  # 9.6 или 3.2 мм
                                d_voxels = max(1, int(brick_height_mm / voxel_size))  # Количество вокселей по высоте
                                if (x + w <= nx and y + h <= ny and z + d_voxels <= nz and
                                    all(voxel_array[z + dz, y + dy, x + dx] and not occupied[z + dz, y + dy, x + dx]
                                        for dz in range(d_voxels) for dy in range(h) for dx in range(w))):
                                    color = LEGO_COLORS[(x + y + z) % len(LEGO_COLORS)] if use_colors else "#FFFFFF"
                                    all_cubes.append((x, y, z, w, h, d, color, brick_type))
                                    for dz in range(d_voxels):
                                        for dy in range(h):
                                            for dx in range(w):
                                                occupied[z + dz, y + dy, x + dx] = True
                                    break

            logging.info(f"Placement completed: {len(all_cubes)} bricks")
            return all_cubes

import os