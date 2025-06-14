import logging
import os
from typing import List, Tuple
import trimesh
import numpy as np
from src.config.config import STUD_SIZE, get_brick_height

def scale_cube(cube: Tuple[float, float, float, int, int, int, str, str]) -> Tuple[float, float, float]:
    """
    Вычисляет координаты центра куба с учетом его размеров и масштаба.
    """
    x, y, z, w, h, d, _, brick_type = cube
    if w <= 0 or h <= 0 or d <= 0:
        raise ValueError(f"Invalid cube dimensions: w={w}, h={h}, d={d} must be positive")
    brick_height = get_brick_height(brick_type)
    return (
        x * STUD_SIZE + w * STUD_SIZE / 2,
        y * STUD_SIZE + h * STUD_SIZE / 2,
        z * brick_height + d * brick_height / 2
    )

def export_voxelized_stl(voxel_array: np.ndarray, output_path: str) -> None:
    if voxel_array is None or not np.any(voxel_array):
        logging.warning("No voxel data to export to STL")
        return

    logging.info(f"Exporting voxelized model to STL: {output_path}")
    try:
        scene = trimesh.Scene()
        for z in range(voxel_array.shape[0]):
            for y in range(voxel_array.shape[1]):
                for x in range(voxel_array.shape[2]):
                    if voxel_array[z, y, x]:
                        box = trimesh.creation.box(extents=(STUD_SIZE, STUD_SIZE, STUD_SIZE))
                        box.apply_translation((x * STUD_SIZE + STUD_SIZE / 2, 
                                              y * STUD_SIZE + STUD_SIZE / 2, 
                                              z * STUD_SIZE + STUD_SIZE / 2))
                        scene.add_geometry(box)
        scene.export(output_path)  # Используем переданный путь
        logging.info(f"Voxelized STL exported: {output_path}")
    except Exception as e:
        logging.error(f"Voxelized STL export error: {e}")
        raise

import trimesh
import numpy as np
from trimesh import boolean

def create_lego_brick(w: int, h: int, d: int, brick_type: str, stud_size: float) -> trimesh.Trimesh:
    """
    Создает модель LEGO-кирпича с шипами и дном с использованием булевых операций.

    Args:
        w, h, d: Ширина, длина и высота в единицах studs.
        brick_type: Тип кирпича (влияет на высоту).
        stud_size: Размер одного шипа (ширина/длина).
    """
    brick_height = get_brick_height(brick_type)
    parts = []

    # Основной корпус
    outer_extents = (w * stud_size, h * stud_size, d * brick_height)
    brick = trimesh.creation.box(extents=outer_extents)

    # Создаем полость
    wall_thickness = stud_size * 0.2
    cavity_extents = (w * stud_size - 2 * wall_thickness, h * stud_size - 2 * wall_thickness, d * brick_height - wall_thickness)
    cavity = trimesh.creation.box(extents=cavity_extents)
    cavity.apply_translation((0, 0, -wall_thickness / 2))  # Смещаем полость вниз

    # Вычитаем полость
    brick = boolean.difference([brick, cavity], engine='blender')

    parts.append(brick)

    # Шипы (studs) сверху
    stud_radius = stud_size * 0.3  # Радиус шипа
    stud_height = stud_size * 0.2  # Высота шипа
    for i in range(w):
        for j in range(h):
            stud = trimesh.creation.cylinder(radius=stud_radius, height=stud_height)
            stud.apply_translation((
                (i + 0.5 - w / 2) * stud_size,  # Центрируем по ширине
                (j + 0.5 - h / 2) * stud_size,  # Центрируем по длине
                (d * brick_height / 2) + (stud_height / 2)  # На верхней грани
            ))
            parts.append(stud)

    # Зажимы (трубки) внутри дна
    clip_radius = stud_size * 0.2  # Радиус трубки-зажима
    clip_height = (d * brick_height - wall_thickness) * 0.8  # Высота трубки
    if w > 1 and h > 1:  # Трубки только для кирпичей больше 1x1
        for i in range(w - 1):
            for j in range(h - 1):
                clip = trimesh.creation.cylinder(radius=clip_radius, height=clip_height)
                clip.apply_translation((
                    (i + 1 - w / 2) * stud_size,  # Позиция между шипами
                    (j + 1 - h / 2) * stud_size,
                    -d * brick_height / 2 + clip_height / 2  # От нижней грани вверх
                ))
                parts.append(clip)

    # Объединяем все части
    brick = trimesh.util.concatenate(parts)

    # Центрируем модель
    brick.apply_translation((w * stud_size / 2, h * stud_size / 2, d * brick_height / 2))
    return brick

def export_unique_bricks_stl(cubes: List[Tuple[float, float, float, int, int, int, str, str]], output_dir: str) -> None:
    if not cubes:
        logging.warning("No cubes to export as unique bricks")
        return

    logging.info(f"Exporting unique LEGO bricks to STL files in: {output_dir}")
    try:
        unique_bricks = {}
        for cube in cubes:
            _, _, _, w, h, d, _, brick_type = cube
            key = (w, h, d, brick_type)
            if key not in unique_bricks:
                unique_bricks[key] = cube
        
        for (w, h, d, brick_type), cube in unique_bricks.items():
            brick = create_lego_brick(w, h, d, brick_type, STUD_SIZE)
            brick_path = os.path.join(output_dir, f"Brick_{w}x{h}x{d}_{brick_type}.stl")  # Полный путь
            brick.export(brick_path)
            logging.info(f"Unique LEGO brick exported: {brick_path}")
    except Exception as e:
        logging.error(f"Unique LEGO bricks STL export error: {e}")
        raise