import logging
from typing import List, Tuple, Optional
import pyvista as pv
import numpy as np
from pyvistaqt import QtInteractor
from src.config.config import (
    BRICK_HEIGHTS, STUD_SIZE,
    LIGHT_INTENSITY_TOP, LIGHT_INTENSITY_SIDES, LIGHT_INTENSITY_AMBIENT, LIGHT_DISTANCE_FACTOR,
    FLOOR_OPACITY, FLOOR_COLOR, FLOOR_EDGE_COLOR, CUBE_EDGE_COLOR, CUBE_OPACITY,
    ORIGINAL_MESH_COLOR, ORIGINAL_MESH_OPACITY, get_brick_height
)

# Константы
FLOOR_Z_POSITION = -10
SCENE_SIZE_MARGIN = 1.5

class SceneRenderer:
    """Класс для рендеринга 3D-сцены."""
    def __init__(self, plotter: QtInteractor):
        self.plotter = plotter
        self.plotter.clear()

    def _calculate_bounds(self, cubes: List[Tuple], scale: float, mesh=None) -> Tuple[float, float, float]:
        """Вычисляет размеры сцены и центр."""
        max_size = 100.0  # Минимальный размер
        center_x, center_y = 0.0, 0.0

        if cubes:
            max_x = max(abs(cube[0]) + cube[3] for cube in cubes) * STUD_SIZE * scale
            max_y = max(abs(cube[1]) + cube[4] for cube in cubes) * STUD_SIZE * scale
            max_size = max(max_x, max_y) * SCENE_SIZE_MARGIN
            center_x = (min(cube[0] * STUD_SIZE * scale for cube in cubes) +
                        max((cube[0] + cube[3]) * STUD_SIZE * scale for cube in cubes)) / 2
            center_y = (min(cube[1] * STUD_SIZE * scale for cube in cubes) +
                        max((cube[1] + cube[4]) * STUD_SIZE * scale for cube in cubes)) / 2
        elif mesh:
            bounds = mesh.bounds
            if bounds.shape == (2, 3):
                max_x = (bounds[1, 0] - bounds[0, 0]) * scale
                max_y = (bounds[1, 1] - bounds[0, 1]) * scale
                max_size = max(max_x, max_y) * SCENE_SIZE_MARGIN
                center_x = (bounds[0, 0] + bounds[1, 0]) / 2 * scale
                center_y = (bounds[0, 1] + bounds[1, 1]) / 2 * scale

        return max_size, center_x, center_y

    def _add_floor(self, size: float):
        """Добавляет пол в сцену."""
        floor = pv.Plane(center=(0, 0, FLOOR_Z_POSITION), i_size=size, j_size=size)
        self.plotter.add_mesh(floor, color=FLOOR_COLOR, show_edges=True, edge_color=FLOOR_EDGE_COLOR,
                              opacity=FLOOR_OPACITY, lighting=False)

    def _add_bricks(self, cubes: List[Tuple], scale: float, center_x: float, center_y: float, highlight_index: Optional[int] = None):
        for i, (x, y, z, w, h, d, color, brick_type) in enumerate(cubes):
            brick_height = get_brick_height(brick_type)  # 9.6 мм или 3.2 мм
            x_mm = x * STUD_SIZE * scale - center_x
            y_mm = y * STUD_SIZE * scale - center_y
            z_mm = z * STUD_SIZE * scale  # z в вокселях по 3.2 мм
            w_mm = w * STUD_SIZE * scale
            h_mm = h * STUD_SIZE * scale
            d_mm = brick_height * scale  # Реальная высота кирпича
            box = pv.Box(bounds=[x_mm, x_mm + w_mm, y_mm, y_mm + h_mm, z_mm, z_mm + d_mm])
            self.plotter.add_mesh(box, color=color, show_edges=True, edge_color=CUBE_EDGE_COLOR,
                                opacity=1.0, lighting=True)

    def _add_original_mesh(self, mesh, scale: float, center_x: float, center_y: float):
        """Добавляет оригинальный меш."""
        faces = np.hstack([np.full((mesh.faces.shape[0], 1), 3, dtype=int), mesh.faces]).ravel()
        mesh_pv = pv.PolyData(mesh.vertices, faces)
        mesh_pv.translate([-center_x, -center_y, 0], inplace=True)
        self.plotter.add_mesh(mesh_pv, color=ORIGINAL_MESH_COLOR, opacity=ORIGINAL_MESH_OPACITY,
                              show_edges=False, lighting=True)

    def _setup_lighting(self, size: float, center_x: float, center_y: float):
        """Настраивает освещение."""
        self.plotter.remove_all_lights()
        light_distance = size * LIGHT_DISTANCE_FACTOR
        light_positions = [
            (center_x, center_y, light_distance),
            (center_x, center_y + light_distance, 0),
            (center_x, center_y - light_distance, 0),
            (center_x - light_distance, center_y, 0),
            (center_x + light_distance, center_y, 0),
        ]
        intensities = [LIGHT_INTENSITY_TOP] + [LIGHT_INTENSITY_SIDES] * 4
        for pos, intensity in zip(light_positions, intensities):
            self.plotter.add_light(pv.Light(position=pos, color="white", intensity=intensity,
                                            positional=True, show_actor=False))
        self.plotter.add_light(pv.Light(color="white", intensity=LIGHT_INTENSITY_AMBIENT,
                                        positional=False, show_actor=False))

    def render(self, cubes: List[Tuple], scale: float = 1.0, mesh=None, show_original: bool = False, 
                highlight_index: Optional[int] = None, voxel_size: float = STUD_SIZE):
            self.plotter.clear()
            global STUD_SIZE
            STUD_SIZE = voxel_size  # Обновляем глобальную константу
            size, center_x, center_y = self._calculate_bounds(cubes, scale, mesh)
            self._add_floor(size)
            if cubes and not show_original:
                self._add_bricks(cubes, scale, center_x, center_y, highlight_index)
            elif mesh and show_original:
                self._add_original_mesh(mesh, scale, center_x, center_y)
            self._setup_lighting(size, center_x, center_y)
            self.plotter.reset_camera()
            logging.debug(f"Scene rendered: {len(cubes)} cubes, scale={scale}, voxel_size={voxel_size}")

def toggle_model_view(app, show_original: bool):
    """Переключает вид модели."""
    app.show_original = show_original
    update_preview(app)
    logging.info(f"View switched to {'original' if show_original else 'voxel'}")

def update_preview(app):
    """Обновляет превью сцены."""   
    renderer = SceneRenderer(app.plotter)
    renderer.render(app.cubes, 1.0, app.original_mesh, app.show_original)
    logging.debug(f"Preview updated: show_original={app.show_original}, cubes={len(app.cubes)}")