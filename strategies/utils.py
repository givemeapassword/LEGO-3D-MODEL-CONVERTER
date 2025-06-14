from numba import njit
import numpy as np
from numba.typed import List as NumbaList
from typing import List, Tuple

import numpy as np
from numba import njit
from src.config.config import MIN_OVERLAP, STUD_SIZE, BRICK_HEIGHTS

@njit(cache=True)
def can_place_brick(x: int, y: int, z: int, w: int, h: int, d: int, voxel_array: np.ndarray, 
                    support_array: np.ndarray, allow_top_layer: bool) -> bool:
    depth, height, width = voxel_array.shape
    
    # Проверка границ
    if (x < 0 or y < 0 or z < 0 or 
        x + w > width or y + h > height or z + d > depth):
        return False
    
    # Проверка пересечения с уже занятыми вокселями
    brick_region = voxel_array[z:z+d, y:y+h, x:x+w]
    if not np.all(brick_region):  # Убедиться, что кирпич полностью заполняет область
        return False
    
    # Проверка поддержки снизу (классическая)
    if z == 0:
        return True
    below = support_array[z-1, y:y+h, x:x+w]
    if below.shape != (h, w) or np.all(~below):
        # Нет полной поддержки снизу, проверяем асимметричные соединения
        overlap_support = False
        
        # Проверяем соседей по сторонам (x, y)
        for dz in range(max(0, z-1), min(depth, z+d)):
            # Слева (x-1)
            if x > 0:
                left = support_array[dz, y:y+h, x-1]
                if left.shape == (h,) and np.sum(left) * STUD_SIZE >= MIN_OVERLAP:
                    overlap_support = True
                    break
            # Справа (x+w)
            if x + w < width:
                right = support_array[dz, y:y+h, x+w]
                if right.shape == (h,) and np.sum(right) * STUD_SIZE >= MIN_OVERLAP:
                    overlap_support = True
                    break
            # Спереди (y-1)
            if y > 0:
                front = support_array[dz, y-1, x:x+w]
                if front.shape == (w,) and np.sum(front) * STUD_SIZE >= MIN_OVERLAP:
                    overlap_support = True
                    break
            # Сзади (y+h)
            if y + h < height:
                back = support_array[dz, y+h, x:x+w]
                if back.shape == (w,) and np.sum(back) * STUD_SIZE >= MIN_OVERLAP:
                    overlap_support = True
                    break
        
        if not overlap_support and not allow_top_layer:
            return False
    
    # Проверка верхнего слоя (если разрешено)
    if z + d == depth and not allow_top_layer:
        return False
    
    return True

@njit(cache=True)
def place_brick(x: int, y: int, z: int, w: int, h: int, d: int, voxel_array: np.ndarray, support_array: np.ndarray):
    voxel_array[z:z+d, y:y+h, x:x+w] = False
    support_array[z:z+d, y:y+h, x:x+w] = True

@njit(cache=True)
def find_next_voxel(voxel_array: np.ndarray) -> Tuple[int, int, int, bool]:
    depth, height, width = voxel_array.shape
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if voxel_array[z, y, x]:
                    return x, y, z, True
    return 0, 0, 0, False

@njit
def place_bricks_on_layer_fast(z: int, voxel_array: np.ndarray, support_array: np.ndarray, 
                               allowed_sizes: NumbaList, allow_top_layer: bool = False) -> List[Tuple[int, int, int, int, int, int, str]]:
    cubes = []
    for y in range(voxel_array.shape[1]):
        for x in range(voxel_array.shape[2]):
            if voxel_array[z, y, x]:
                for w, h, d, t in allowed_sizes:
                    if can_place_brick(x, y, z, w, h, d, voxel_array, support_array, allow_top_layer):
                        place_brick(x, y, z, w, h, d, voxel_array, support_array)
                        cubes.append((x, y, z, w, h, d, t))
                        break
                    else:
                        # Добавьте отладочный вывод через logging (вне njit для простоты)
                        print(f"Cannot place brick at ({x}, {y}, {z}) with size ({w}, {h}, {d})")
    return cubes