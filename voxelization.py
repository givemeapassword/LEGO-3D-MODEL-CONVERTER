import numpy as np
import trimesh
import logging
from typing import List, Tuple
from src.config.config import DEFAULT_RADIUS, EPSILON, MAX_ITER_MULTIPLIER, MAX_RADIUS, MIN_RADIUS, RADIUS_FRACTION, STUD_SIZE

VOXELIZATION_METHOD = "subdivide"
MAX_ITER_EXCEEDED_MSG = "max_iter exceeded"

def validate_voxelization_inputs(mesh: trimesh.Trimesh, max_depth: int, voxel_size: float) -> None:
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Параметр 'mesh' должен быть объектом trimesh.Trimesh")
    if max_depth <= 0:
        raise ValueError("max_depth должен быть положительным числом")
    if voxel_size <= 0:
        raise ValueError("voxel_size должен быть положительным числом")

def voxelize_with_retry(
    mesh: trimesh.Trimesh,
    voxel_size: float,
    max_depth: int,
    method: str = VOXELIZATION_METHOD
) -> trimesh.voxel.VoxelGrid:
    try:
        return mesh.voxelized(pitch=voxel_size, method=method, max_iter=max_depth)
    except ValueError as ve:
        if MAX_ITER_EXCEEDED_MSG in str(ve):
            new_max_depth = max_depth * MAX_ITER_MULTIPLIER
            logging.warning(f"Превышено max_iter={max_depth}. Повтор с max_iter={new_max_depth}.")
            return mesh.voxelized(pitch=voxel_size, method=method, max_iter=new_max_depth)
        raise

def analyze_model_regions(mesh: trimesh.Trimesh, radius: float = DEFAULT_RADIUS, 
                         min_region_size: float = 10.0) -> List[Tuple[np.ndarray, float]]:
    bounds = mesh.bounds
    max_dim = max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1], bounds[1][2] - bounds[0][2])
    dynamic_radius = max(MIN_RADIUS, min(MAX_RADIUS, max_dim * RADIUS_FRACTION))
    
    logging.info(f"Dynamic radius: {dynamic_radius:.2f} (max_dim={max_dim:.2f})")
    curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, dynamic_radius)
    min_curv, max_curv = np.min(curvature), np.max(curvature)
    
    if max_curv - min_curv < EPSILON:
        return [(mesh.vertices, curvature.mean())]
    
    normalized_curvature = (curvature - min_curv) / (max_curv - min_curv + EPSILON)
    regions = []
    # Перекрытие регионов для устранения вертикальных разрывов
    high_curv_vertices = mesh.vertices[normalized_curvature > 0.2]  # Снижаем порог
    low_curv_vertices = mesh.vertices[normalized_curvature <= 0.4]  # Увеличиваем перекрытие
    
    if len(high_curv_vertices) > 0:
        high_bounds = trimesh.bounds.corners([high_curv_vertices.min(axis=0), high_curv_vertices.max(axis=0)])
        if np.all(high_bounds[1] - high_bounds[0] >= min_region_size):
            regions.append((high_curv_vertices, normalized_curvature[normalized_curvature > 0.2].mean()))
    
    if len(low_curv_vertices) > 0:
        low_bounds = trimesh.bounds.corners([low_curv_vertices.min(axis=0), low_curv_vertices.max(axis=0)])
        if np.all(low_bounds[1] - low_bounds[0] >= min_region_size):
            regions.append((low_curv_vertices, normalized_curvature[normalized_curvature <= 0.4].mean()))
    
    return regions if regions else [(mesh.vertices, curvature.mean())]

def voxelize_region(mesh: trimesh.Trimesh, region_vertices: np.ndarray, base_voxel_size: float, 
                    curvature_mean: float, max_depth: int) -> Tuple[trimesh.voxel.VoxelGrid, float]:
    # Фиксированный размер вокселя, соответствующий LEGO
    voxel_size = base_voxel_size  # Например, 8 мм
    
    # Увеличиваем детализацию меша в регионах с высокой кривизной
    subdivision_size = voxel_size / (1 + curvature_mean)  # Чем выше кривизна, тем мельче разбиение
    subdivision_size = max(voxel_size / 2, min(voxel_size, subdivision_size))  # Ограничиваем
    
    region_mesh = mesh.copy()
    region_mesh.vertices = region_vertices
    region_mesh = region_mesh.subdivide_to_size(subdivision_size)
    
    grid = voxelize_with_retry(region_mesh, voxel_size, max_depth)
    return grid, voxel_size

def merge_voxel_grids(voxel_grids: List[Tuple[trimesh.voxel.VoxelGrid, float]], mesh_bounds: np.ndarray) -> Tuple[np.ndarray, float]:
    min_bounds, max_bounds = mesh_bounds
    effective_voxel_size = min(voxel_size for _, voxel_size in voxel_grids)
    total_shape = np.ceil((max_bounds - min_bounds) / effective_voxel_size).astype(int)
    merged_array = np.zeros(total_shape, dtype=bool)
    
    for grid, voxel_size in voxel_grids:
        scale_factor = voxel_size / effective_voxel_size
        start = np.zeros(3, dtype=int)
        end = (np.array(grid.matrix.shape) * scale_factor).astype(int)
        valid_start = np.maximum(start, 0)
        valid_end = np.minimum(end, total_shape)
        grid_slice = grid.matrix[
            max(0, valid_start[0] - start[0]):grid.matrix.shape[0] - (end[0] - valid_end[0]),
            max(0, valid_start[1] - start[1]):grid.matrix.shape[1] - (end[1] - valid_end[1]),
            max(0, valid_start[2] - start[2]):grid.matrix.shape[2] - (end[2] - valid_end[2])
        ]
        merged_array[
            valid_start[0]:valid_end[0],
            valid_start[1]:valid_end[1],
            valid_start[2]:valid_end[2]
        ] |= grid_slice
    
    return merged_array, effective_voxel_size

def adaptive_voxelization(
    mesh: trimesh.Trimesh,
    max_depth: int = 10,
    voxel_size: float = STUD_SIZE,
    curvature_based: bool = False,
    radius: float = DEFAULT_RADIUS
) -> trimesh.voxel.VoxelGrid:
    validate_voxelization_inputs(mesh, max_depth, voxel_size)
    logging.info(f"Starting uniform voxelization: max_depth={max_depth}, voxel_size={voxel_size}")
    voxel_grid = voxelize_with_retry(mesh, voxel_size, max_depth)
    logging.info(f"Voxelization completed: {len(voxel_grid.points)} voxels")
    return voxel_grid

def voxel_grid_to_numpy(voxel_grid: trimesh.voxel.VoxelGrid) -> np.ndarray:
    if not isinstance(voxel_grid, trimesh.voxel.VoxelGrid):
        raise ValueError("Параметр 'voxel_grid' должен быть объектом trimesh.voxel.VoxelGrid")
    return voxel_grid.matrix.astype(bool)