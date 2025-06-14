import logging
from typing import Tuple, Optional
from PyQt5.QtCore import Qt
import numpy as np
import pyvista as pv
from src.config.config import CAMERA_DISTANCE_FACTOR,VIEW_CONFIGS

def get_model_bounds(app) -> Tuple[float, float, float, float, float, float]:
    """Compute the model bounds from cubes or the original mesh, handling various bounds formats."""
    if app.cubes:
        combined_mesh = pv.PolyData()
        for cube in app.cubes:
            if isinstance(cube, tuple) and len(cube) >= 6:  # Формат (x, y, z, w, h, d, ...)
                x, y, z, w, h, d = cube[:6]
                center = (x + w/2, y + h/2, z + d/2)
            elif isinstance(cube, dict):  # Формат словаря
                x, y, z = cube["center"]
                w, h, d = cube["width"], cube["height"], cube["depth"]
                center = (x + w/2, y + h/2, z + d/2)
            elif hasattr(cube, "center") and hasattr(cube, "x_length"):  # Объект pv.Cube
                x, y, z = cube.center
                w, h, d = cube.x_length, cube.y_length, cube.z_length
                center = (x, y, z)
            else:
                logging.warning(f"Unknown cube format: {cube}")
                continue
            mesh = pv.Cube(center=center, x_length=w, y_length=h, z_length=d)
            combined_mesh += mesh
        bounds = combined_mesh.bounds
    elif app.original_mesh:
        bounds = app.original_mesh.bounds
    else:
        bounds = pv.Cube(center=(0, 0, 0), x_length=10, y_length=10, z_length=10).bounds

    # Остальной код остается без изменений
    if isinstance(bounds, np.ndarray):
        if bounds.ndim == 2 and bounds.shape == (2, 3):
            full_bounds = [
                bounds[0][0], bounds[1][0],
                bounds[0][1], bounds[1][1],
                bounds[0][2], bounds[1][2]
            ]
        else:
            logging.error(f"Unexpected bounds format: {bounds}")
            full_bounds = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
    elif isinstance(bounds, (list, tuple)) and len(bounds) == 6:
        full_bounds = list(bounds)
    else:
        logging.warning(f"Model bounds incomplete: {bounds}. Expanding to 3D.")
        full_bounds = [0.0] * 6
        for i in range(min(len(bounds), 6)):
            full_bounds[i] = bounds[i]

    for i in range(0, 6, 2):
        if full_bounds[i] == full_bounds[i + 1]:
            full_bounds[i] -= 0.1
            full_bounds[i + 1] += 0.1
            logging.debug(f"Expanded axis {i//2}: {full_bounds[i]} to {full_bounds[i+1]}")

    return tuple(full_bounds)

def calculate_camera_position(bounds: Tuple[float, float, float, float, float, float], 
                              direction: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Вычисляет позицию камеры на основе направления и границ модели."""
    x_range, y_range, z_range = bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
    max_dimension = max(x_range, y_range, z_range)
    camera_distance = max_dimension * CAMERA_DISTANCE_FACTOR
    center = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2)
    return (
        center[0] + direction[0] * camera_distance,
        center[1] + direction[1] * camera_distance,
        center[2] + direction[2] * camera_distance
    )

def set_view(app, view: str, 
             camera_pos: Optional[Tuple[float, float, float]] = None,
             camera_focal_point: Optional[Tuple[float, float, float]] = None,
             camera_view_up: Optional[Tuple[float, float, float]] = None) -> None:
    """Устанавливает вид камеры с анимацией."""
    logging.debug(f"Setting view: {view}")
    bounds = get_model_bounds(app)
    center = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2)
    current_pos, current_focal, current_up = app.plotter.camera_position

    if view == "custom" and camera_pos is not None and camera_focal_point is not None and camera_view_up is not None:
        app.animate_camera(current_pos, camera_pos, current_focal, camera_focal_point, current_up, camera_view_up)
        logging.debug(f"Custom view set: pos={camera_pos}, focal={camera_focal_point}, up={camera_view_up}")
    elif view in VIEW_CONFIGS:
        direction, view_up = VIEW_CONFIGS[view]
        target_pos = calculate_camera_position(bounds, direction)
        app.animate_camera(current_pos, target_pos, current_focal, center, current_up, view_up)
        logging.debug(f"{view} view set: pos={target_pos}, up={view_up}")
    elif view.startswith("iso-"):  # Обработка изометрических видов
        parts = view.split("-")[1:]  # Например, ["top", "front", "right"]
        direction = [0, 0, 0]
        for part in parts:
            if part in VIEW_CONFIGS:
                dir_component, _ = VIEW_CONFIGS[part]
                direction = [d + dc for d, dc in zip(direction, dir_component)]
        direction = tuple(d / np.linalg.norm(direction) for d in direction)  # Нормализуем
        view_up = (0, 0, 1)  # По умолчанию вверх по Z
        target_pos = calculate_camera_position(bounds, direction)
        app.animate_camera(current_pos, target_pos, current_focal, center, current_up, view_up)
        logging.debug(f"Isometric view set: {view}, pos={target_pos}, up={view_up}")
    elif view.startswith("edge-"):  # Обработка видов ребер
        parts = view.split("-")[1:]  # Например, ["top", "front"]
        direction = [0, 0, 0]
        for part in parts:
            if part in VIEW_CONFIGS:
                dir_component, _ = VIEW_CONFIGS[part]
                direction = [d + dc for d, dc in zip(direction, dir_component)]
        direction = tuple(d / np.linalg.norm(direction) for d in direction)  # Нормализуем
        view_up = (0, 0, 1)  # По умолчанию вверх по Z
        target_pos = calculate_camera_position(bounds, direction)
        app.animate_camera(current_pos, target_pos, current_focal, center, current_up, view_up)
        logging.debug(f"Edge view set: {view}, pos={target_pos}, up={view_up}")
    else:
        logging.warning(f"Unknown view requested: {view}")

class ModelInteractionHandler:
    """Обработчик взаимодействия с моделью через мышь."""
    DRAG_SPEED = 0.1   # Скорость перетаскивания
    ROTATE_SPEED = 0.01  # Скорость вращения
    ZOOM_IN_FACTOR = 1.1  # Увеличение при прокрутке вверх
    ZOOM_OUT_FACTOR = 0.9  # Уменьшение при прокрутке вниз

    @staticmethod
    def handle_mouse_press(app, event) -> bool:
        """Обрабатывает нажатие мыши."""
        if event.button() == Qt.LeftButton:
            app.last_mouse_pos = event.pos()
            if app.is_dragging:
                app.is_rotating = False
            elif app.is_rotating:
                app.is_dragging = False
        return True

    @staticmethod
    def handle_mouse_move(app, event) -> bool:
        """Обрабатывает движение мыши."""
        if not (event.buttons() & Qt.LeftButton):
            return False
        delta = event.pos() - app.last_mouse_pos
        app.last_mouse_pos = event.pos()

        if app.is_dragging:
            dx = delta.x() * ModelInteractionHandler.DRAG_SPEED
            dy = delta.y() * ModelInteractionHandler.DRAG_SPEED
            pos = app.plotter.camera_position[0]
            app.plotter.camera_position = ((pos[0] - dx, pos[1] + dy, pos[2]), *app.plotter.camera_position[1:])
            logging.debug(f"Dragging: dx={dx:.2f}, dy={dy:.2f}")
        elif app.is_rotating:
            dx = delta.x() * ModelInteractionHandler.ROTATE_SPEED
            dy = delta.y() * ModelInteractionHandler.ROTATE_SPEED
            up = app.plotter.camera_position[2]
            app.plotter.camera_position = (*app.plotter.camera_position[:2], (up[0] + dx, up[1] + dy, up[2]))
            logging.debug(f"Rotating: dx={dx:.2f}, dy={dy:.2f}")
        
        app.plotter.update()
        return True

    @staticmethod
    def handle_mouse_release(app, event) -> bool:
        """Обрабатывает отпускание мыши."""
        if event.button() == Qt.LeftButton and not (app.is_dragging or app.is_rotating):
            app.is_dragging = False
            app.is_rotating = False
        return True

    @staticmethod
    def handle_wheel(app, event) -> bool:
        """Обрабатывает прокрутку колеса мыши для зума."""
        delta = event.angleDelta().y()
        zoom_factor = ModelInteractionHandler.ZOOM_IN_FACTOR if delta > 0 else ModelInteractionHandler.ZOOM_OUT_FACTOR
        pos = app.plotter.camera_position[0]
        app.plotter.camera_position = ((pos[0] * zoom_factor, pos[1] * zoom_factor, pos[2] * zoom_factor), *app.plotter.camera_position[1:])
        app.plotter.update()
        logging.debug(f"Zooming: factor={zoom_factor:.2f}")
        return True

def handle_interaction(app, source, event) -> bool:
    """Обрабатывает события мыши для взаимодействия с моделью."""
    if source != app.plotter.interactor:
        return False
    
    event_type = event.type()
    if event_type == event.MouseButtonPress:
        return ModelInteractionHandler.handle_mouse_press(app, event)
    elif event_type == event.MouseMove:
        return ModelInteractionHandler.handle_mouse_move(app, event)
    elif event_type == event.MouseButtonRelease:
        return ModelInteractionHandler.handle_mouse_release(app, event)
    elif event_type == event.Wheel:
        return ModelInteractionHandler.handle_wheel(app, event)
    return False