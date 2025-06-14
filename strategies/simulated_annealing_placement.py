import numpy as np
import logging
import concurrent.futures
from numba.typed import List as NumbaList
from typing import List, Tuple
from scipy.ndimage import label
from src.config.config import LEGO_COLORS
from src.strategies.base import PlacementStrategy
from src.strategies.utils import place_bricks_on_layer_fast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimulatedAnnealingPlacementStrategy(PlacementStrategy):
    def place_bricks(self, voxel_array, use_colors, allowed_sizes, allow_top_layer=False, progress_callback=None,
                        initial_temp: float = 1000.0, min_temp: float = 1.0, max_iterations: int = 100, brick_type=None):
            voxel_copy = voxel_array.copy()
            support_array = np.zeros_like(voxel_array, dtype=bool)
            total_voxels = np.sum(voxel_array)
            temperature = initial_temp
            cubes = []
            layer_cache = {}

            def cost_function(cubes_list: List[Tuple]) -> float:
                # Текущее покрытие
                coverage = sum(w * h * d for _, _, _, w, h, d, _ in cubes_list)
                base_cost = -len(cubes_list) - coverage / total_voxels

                # Построим воксельный массив текущих кирпичей
                brick_array = np.zeros_like(voxel_array, dtype=bool)
                for x, y, z, w, h, d, _ in cubes_list:
                    brick_array[z:z+d, y:y+h, x:x+w] = True
                
                # Анализ связности
                labeled_array, num_clusters = label(brick_array)
                fragmentation_penalty = num_clusters * 5  # Штраф за каждый лишний кластер (коэффициент подстройки)
                
                return base_cost + fragmentation_penalty

            best_cubes = self._initial_greedy_placement(voxel_copy, support_array, allowed_sizes, allow_top_layer, layer_cache)
            best_cost = cost_function(best_cubes)
            current_voxel = voxel_copy.copy()
            current_support = support_array.copy()

            iteration = 0
            while temperature > min_temp and iteration < max_iterations:
                if progress_callback and progress_callback(sum(w * h * d for _, _, _, w, h, d, _ in best_cubes) / total_voxels):  # Проверка остановки
                    return [(x, y, z, w, h, d, np.random.choice(LEGO_COLORS) if use_colors else "#000000", brick_type) for x, y, z, w, h, d, _ in best_cubes]
                iteration += 1
                coverage = sum(w * h * d for _, _, _, w, h, d, _ in best_cubes) / total_voxels
                action_probs = {"remove": min(0.5, coverage), "add": 0.3, "replace": 1 - min(0.5, coverage) - 0.3}

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self._perturb_solution, best_cubes, current_voxel.copy(), 
                                            current_support.copy(), allowed_sizes, allow_top_layer, layer_cache, 
                                            np.random.choice(list(action_probs.keys()), p=list(action_probs.values())))
                            for _ in range(3)]
                    new_cubes_list = [f.result() for f in concurrent.futures.as_completed(futures)]

                new_cubes = min(new_cubes_list, key=cost_function)
                new_cost = cost_function(new_cubes)

                delta_cost = new_cost - best_cost
                if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / temperature):
                    best_cubes = new_cubes
                    best_cost = new_cost
                    voxel_copy[:] = current_voxel
                    support_array[:] = current_support

                best_cubes = self._local_optimization(best_cubes, voxel_copy, support_array, allowed_sizes, allow_top_layer, layer_cache)

                temperature = initial_temp / (1 + iteration * 0.1)
                processed_voxels = sum(w * h * d for _, _, _, w, h, d, _ in best_cubes)
                logging.info("SA: Iter %d, Temp %.2f, Cost %.2f, Clusters %d, Progress %.2f%%" % 
                            (iteration, temperature, best_cost, label(np.zeros_like(voxel_array, dtype=bool))[1], 
                            processed_voxels / total_voxels * 100))
                if progress_callback:
                    progress_callback(processed_voxels / total_voxels)

            for cube in best_cubes:
                x, y, z, w, h, d, _ = cube
                color = np.random.choice(LEGO_COLORS) if use_colors else "#000000"
                cubes.append((x, y, z, w, h, d, color, brick_type))
            return cubes

    def _initial_greedy_placement(self, voxel_array: np.ndarray, support_array: np.ndarray, 
                                  allowed_sizes: NumbaList, allow_top_layer: bool, layer_cache: dict) -> List[Tuple]:
        cubes = []
        for z in range(voxel_array.shape[0]):
            if z not in layer_cache:
                layer_cubes = place_bricks_on_layer_fast(z, voxel_array, support_array, allowed_sizes, allow_top_layer)
                layer_cache[z] = [(x, y, z, w, h, d) for x, y, z, w, h, d in layer_cubes]
            cubes.extend((x, y, z, w, h, d, "#000000") for x, y, z, w, h, d in layer_cache[z])
        return cubes

    def _perturb_solution(self, cubes: List[Tuple], voxel_array: np.ndarray, support_array: np.ndarray, 
                          allowed_sizes: NumbaList, allow_top_layer: bool, layer_cache: dict, action: str) -> List[Tuple]:
        new_cubes = cubes.copy()
        total_coverage = sum(w * h * d for _, _, _, w, h, d, _ in new_cubes)

        # Идея 3: Ограничение пертурбаций
        if total_coverage >= np.sum(voxel_array) and action != "remove":
            return new_cubes

        if action == "remove" and new_cubes:
            new_cubes.pop(np.random.randint(len(new_cubes)))
        elif action == "add":
            z = np.random.randint(voxel_array.shape[0])
            if z not in layer_cache:
                layer_cache[z] = place_bricks_on_layer_fast(z, voxel_array, support_array, allowed_sizes, allow_top_layer)
            if layer_cache[z]:
                x, y, z, w, h, d = layer_cache[z][0]
                new_cubes.append((x, y, z, w, h, d, "#000000"))
        elif action == "replace" and new_cubes:
            idx = np.random.randint(len(new_cubes))
            old_cube = new_cubes.pop(idx)
            z = old_cube[2]
            if z not in layer_cache:
                layer_cache[z] = place_bricks_on_layer_fast(z, voxel_array, support_array, allowed_sizes, allow_top_layer)
            if layer_cache[z]:
                x, y, z, w, h, d = layer_cache[z][0]
                new_cubes.append((x, y, z, w, h, d, "#000000"))

        return new_cubes

    def _local_optimization(self, cubes: List[Tuple], voxel_array: np.ndarray, support_array: np.ndarray, 
                            allowed_sizes: NumbaList, allow_top_layer: bool, layer_cache: dict) -> List[Tuple]:
        """
        Локальная жадная доработка размещения кирпичей.
        
        Args:
            cubes: Текущий список размещённых кубов (x, y, z, w, h, d, color, brick_type).
            voxel_array: Воксельный массив модели.
            support_array: Массив поддержки (True - поддерживаемый воксель).
            allowed_sizes: Список разрешённых размеров кирпичей (w, h, d, brick_type).
            allow_top_layer: Разрешить размещение на верхнем слое с боковой поддержкой.
            layer_cache: Кэш слоёв для ускорения.
        
        Returns:
            Обновлённый список кубов с локальными улучшениями.
        """
        optimized_cubes = cubes.copy()
        total_voxels = np.sum(voxel_array)
        current_coverage = sum(w * h * d for _, _, _, w, h, d, _, _ in optimized_cubes)

        for z in range(voxel_array.shape[0]):
            if z not in layer_cache:
                # Используем place_bricks_on_layer_fast для получения новых кубов
                layer_cubes = place_bricks_on_layer_fast(z, voxel_array, support_array, allowed_sizes, allow_top_layer)
                layer_cache[z] = [(x, y, z, w, h, d, t) for x, y, z, w, h, d, t in layer_cubes]
            
            # Добавляем только те кубы, которые увеличивают покрытие и не перекрываются
            for x, y, z, w, h, d, brick_type in layer_cache[z]:
                if current_coverage < total_voxels:
                    # Проверяем, не перекрывается ли новый куб с существующими
                    overlap = False
                    for existing_cube in optimized_cubes:
                        ex, ey, ez, ew, eh, ed, _, _ = existing_cube
                        if (x < ex + ew and x + w > ex and
                            y < ey + eh and y + h > ey and
                            z < ez + ed and z + d > ez):
                            overlap = True
                            break
                    
                    if not overlap:
                        optimized_cubes.append((x, y, z, w, h, d, "#000000", brick_type))
                        current_coverage += w * h * d
        
        return optimized_cubes