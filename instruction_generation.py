from sklearn.cluster import DBSCAN
import numpy as np
from scipy.ndimage import label
import logging
from numba import njit
from multiprocessing import Pool
import os
import io
import shutil
from PIL import Image
import pyvista as pv
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from typing import List, Tuple
from src.config.config import (
    STUD_SIZE, PDF_PAGE_SIZE, TEMP_IMAGE_DIR, RENDER_LIGHT_POSITION, get_brick_height
)

# --- Константы для PDF ---
PDF_PAGE_SIZE = (842, 595)  # Альбомная ориентация (A4 горизонтально)
PDF_HEADER_Y = 550  # Номер шага слева сверху
PDF_HEADER_FONT_SIZE = 24
PDF_DESC_Y = 50  # Список кубиков слева снизу
PDF_DESC_FONT_SIZE = 12
PDF_STEP_IMAGE_X = 400  # Картинка шага справа сверху
PDF_STEP_IMAGE_Y = 300
PDF_STEP_IMAGE_WIDTH = 400
PDF_STEP_IMAGE_HEIGHT = 250
PDF_FULL_MODEL_X = 650  # Полная модель справа снизу
PDF_FULL_MODEL_Y = 50
PDF_FULL_MODEL_SIZE = 150
MAX_BRICKS_PER_STEP = 50

# --- Генерация инструкций ---

@njit
def _generate_instructions_for_component_numba(occupied, cubes, shape):
    instructions = np.zeros((len(cubes), 7), dtype=np.int32)
    instruction_count = 0
    for i in range(len(cubes)):
        x, y, z, w, h, d = cubes[i]
        overlap = False
        for dz in range(z, min(z + d, shape[0])):
            for dy in range(y, min(y + h, shape[1])):
                for dx in range(x, min(x + w, shape[2])):
                    if occupied[dz, dy, dx]:
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                break
        if not overlap:
            instructions[instruction_count] = [x, y, z, w, h, d, 0]
            instruction_count += 1
            for dz in range(z, min(z + d, shape[0])):
                for dy in range(y, min(y + h, shape[1])):
                    for dx in range(x, min(x + w, shape[2])):
                        occupied[dz, dy, dx] = True
    return instructions[:instruction_count]

def find_connected_components(voxel_array: np.ndarray, clustering_method: str = 'connected') -> tuple:
    if voxel_array.size == 0 or not np.any(voxel_array):
        logging.warning("Voxel array is empty")
        return np.zeros_like(voxel_array, dtype=int), 0

    if clustering_method == 'connected':
        labeled_array, num_features = label(voxel_array)
    elif clustering_method == 'dbscan':
        coords = np.argwhere(voxel_array)
        if len(coords) == 0:
            logging.warning("No voxels for DBSCAN clustering")
            return np.zeros_like(voxel_array, dtype=int), 0
        clustering = DBSCAN(eps=1.5, min_samples=5).fit(coords)
        labels = clustering.labels_
        num_features = len(set(labels)) - (1 if -1 in labels else 0)
        labeled_array = np.zeros_like(voxel_array, dtype=int)
        for i, lbl in enumerate(labels):
            if lbl != -1:
                labeled_array[tuple(coords[i])] = lbl + 1
    else:
        raise ValueError(f"Unsupported clustering method: {clustering_method}")
    
    logging.info(f"Found {num_features} components")
    return labeled_array, num_features

def generate_instructions_for_component(component_voxels: np.ndarray, cubes: list, progress_callback=None) -> list:
    if not cubes:
        return []
    
    cube_array = np.array([[cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]] 
                          for cube in sorted(cubes, key=lambda cube: cube[2])], dtype=np.int32)
    occupied = np.zeros_like(component_voxels, dtype=np.bool_)
    
    instructions_array = _generate_instructions_for_component_numba(occupied, cube_array, component_voxels.shape)
    
    instructions = []
    cube_dict = {(c[0], c[1], c[2]): (c[6], c[7]) for c in cubes}
    total_cubes = len(instructions_array)
    for i, instr in enumerate(instructions_array):
        x, y, z, w, h, d, _ = instr
        color, brick_type = cube_dict.get((x, y, z), ("gray", "brick"))
        instructions.append((x, y, z, w, h, d, color, brick_type))
        if progress_callback and total_cubes > 0:
            progress_callback((i + 1) / total_cubes)
    
    return instructions

def process_component(args):
    label, labeled_array, cubes_by_label, progress_callback = args
    component_voxels = labeled_array == label
    component_cubes = cubes_by_label.get(label, [])
    
    def component_progress(progress):
        if callable(progress_callback):
            progress_callback((label - 1 + progress) / len(cubes_by_label))
    return generate_instructions_for_component(component_voxels, component_cubes, component_progress)

def generate_instructions(voxel_array: np.ndarray, cubes: list, clustering_method: str = 'connected', 
                         parallel: bool = False, progress_callback=None) -> list:
    labeled_array, num_features = find_connected_components(voxel_array, clustering_method)
    if num_features == 0:
        logging.info("No components to generate instructions")
        return []

    logging.info(f"Generating instructions: method={clustering_method}, components={num_features}")
    cubes_by_label = {label: [] for label in range(1, num_features + 1)}
    for cube in cubes:
        x, y, z = cube[0], cube[1], cube[2]
        if 0 <= z < labeled_array.shape[0] and 0 <= y < labeled_array.shape[1] and 0 <= x < labeled_array.shape[2]:
            label = labeled_array[z, y, x]
            if label > 0:
                cubes_by_label[label].append(cube)
        else:
            logging.warning(f"Cube {cube[:3]} outside voxel array")

    instructions = []
    if parallel and num_features > 4:
        with Pool() as pool:
            def total_progress(progress):
                if callable(progress_callback):
                    progress_callback(progress)
            results = pool.map(process_component, [(label, labeled_array, cubes_by_label, total_progress) 
                                                  for label in range(1, num_features + 1)])
        instructions = [cube for sublist in results for cube in sublist]
    else:
        total_components = num_features
        for i, label in enumerate(range(1, num_features + 1)):
            component_voxels = labeled_array == label
            component_cubes = cubes_by_label.get(label, [])
            def component_progress(progress):
                if callable(progress_callback) and total_components > 0:
                    progress_callback((i + progress) / total_components)
            instructions.extend(generate_instructions_for_component(component_voxels, component_cubes, component_progress))
    logging.debug(f"Instruction details: cubes={len(cubes)}, clustering={clustering_method}")
    logging.info(f"Instructions generated: {len(instructions)} steps")
    return instructions

# --- Генерация PDF ---

def create_plotter(window_size: tuple, off_screen: bool = True) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    plotter.set_background("white")
    plotter.add_light(pv.Light(position=RENDER_LIGHT_POSITION))
    plotter.enable_parallel_projection()
    return plotter

def render_step(cubes: list, current_step_index: int, output_path: str) -> None:
    try:
        plotter = create_plotter(window_size=(800, 500))
        total_cubes = len(cubes)
        for i, cube in enumerate(cubes):  # Рендерим все кубики до конца шага
            x, y, z, w, h, d, color, brick_type = cube
            brick_height = get_brick_height(brick_type)
            bounds = [
                x * STUD_SIZE, (x + w) * STUD_SIZE,
                y * STUD_SIZE, (y + h) * STUD_SIZE,
                z * brick_height, (z + d) * brick_height
            ]
            box = pv.Box(bounds=bounds)
            opacity = 1.0
            # Подсвечиваем только кубики текущего шага как яркие
            is_new = i >= (total_cubes - len(steps[current_step_index])) if steps else False
            if isinstance(color, str) and color.startswith("#"):
                adjusted_color = color if is_new else f"{color}80"
            else:
                r, g, b = [int(c * 255) for c in color[:3]] if isinstance(color, (tuple, list)) else (128, 128, 128)
                adjusted_color = f"#{r:02x}{g:02x}{b:02x}" if is_new else f"#{r:02x}{g:02x}{b:02x}80"
            plotter.add_mesh(box, color=adjusted_color, opacity=opacity, show_edges=True)
        
        plotter.camera.zoom(1.5)
        plotter.reset_camera()
        plotter.screenshot(output_path)
        plotter.close()
        if os.path.exists(output_path):
            logging.info(f"Step image saved: {output_path}")
        else:
            logging.error(f"Failed to save step image: {output_path}")
    except Exception as e:
        logging.error(f"Ошибка рендеринга шага {current_step_index + 1}: {e}")

def render_full_model(cubes: list, output_buffer: io.BytesIO) -> None:
    try:
        plotter = create_plotter(window_size=(150, 150))
        for cube in cubes:
            x, y, z, w, h, d, color, brick_type = cube
            brick_height = get_brick_height(brick_type)
            bounds = [
                x * STUD_SIZE, (x + w) * STUD_SIZE,
                y * STUD_SIZE, (y + h) * STUD_SIZE,
                z * brick_height, (z + d) * brick_height
            ]
            box = pv.Box(bounds=bounds)
            plotter.add_mesh(box, color=color, opacity=1.0, show_edges=True)
        
        plotter.camera.zoom(1.5)
        plotter.reset_camera()
        img_array = plotter.screenshot(return_img=True)
        plotter.close()
        Image.fromarray(img_array).save(output_buffer, format="PNG")
        output_buffer.seek(0)
    except Exception as e:
        logging.error(f"Ошибка рендеринга полной модели: {e}")

def get_layer_steps(cubes: list, max_bricks_per_step: int = MAX_BRICKS_PER_STEP) -> List[List[Tuple]]:
    layers = {}
    for cube in cubes:
        z = cube[2]
        layers.setdefault(z, []).append(cube)
    
    steps = []
    for z in sorted(layers.keys()):
        layer_cubes = layers[z]
        for i in range(0, len(layer_cubes), max_bricks_per_step):
            steps.append(layer_cubes[i:i + max_bricks_per_step])
    return steps

def generate_brick_icon(w: int, h: int, d: int, color: str, brick_type: str, output_path: str) -> None:
    plotter = create_plotter(window_size=(50, 50))
    brick_height = get_brick_height(brick_type)
    bounds = [0, w * STUD_SIZE, 0, h * STUD_SIZE, 0, d * brick_height]
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, color=color, opacity=1.0, show_edges=True)
    plotter.add_text(f"{w}x{h}x{d}", position="upper_left", font_size=10, color="black")
    plotter.camera.zoom(1.5)
    plotter.reset_camera()
    plotter.screenshot(output_path)
    plotter.close()

def add_parts_list_page(pdf: canvas.Canvas, cubes: List[Tuple[float, float, float, int, int, int, str, str]], 
                       color_names: dict) -> None:
    pdf.setFillColor(HexColor("#FFFFFF"))  # Белый фон
    pdf.rect(0, 0, PDF_PAGE_SIZE[0], PDF_PAGE_SIZE[1], fill=1)
    
    # Заголовок
    pdf.setFillColorRGB(0, 0, 0)  # Черный текст
    pdf.setFont("Helvetica-Bold", PDF_HEADER_FONT_SIZE + 4)
    pdf.drawCentredString(PDF_PAGE_SIZE[0] / 2, PDF_HEADER_Y, "Parts List")

    # Подсчет уникальных кубиков
    brick_counts = {}
    for cube in cubes:
        x, y, z, w, h, d, color, brick_type = cube
        key = (w, h, d, color, brick_type)
        brick_counts[key] = brick_counts.get(key, 0) + 1

    # Рамка для списка
    list_x, list_y, list_width, list_height = 50, 50, 700, 450
    pdf.setFillColor(HexColor("#FFFFFF"))
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf.roundRect(list_x, list_y, list_width, list_height, 10, stroke=1, fill=1)  # Закругленные углы

    # Динамическое масштабирование для одной страницы
    num_items = len(brick_counts)
    max_cols = 4  # До 4 столбцов
    item_height = min(40, list_height // ((num_items // max_cols) + 1))
    item_width = list_width // max_cols
    icon_size = min(30, item_height - 10)

    y_pos = list_y + list_height - item_height
    x_pos = list_x + 10
    col = 0
    pdf.setFont("Helvetica", max(8, min(PDF_DESC_FONT_SIZE, item_height - 10)))  # Минимум 8pt
    pdf.setFillColorRGB(0, 0, 0)  # Черный текст

    for (w, h, d, color, brick_type), count in brick_counts.items():
        brick_icon_path = os.path.join(TEMP_IMAGE_DIR, f"brick_{w}_{h}_{d}_{color}_{brick_type}.png")
        generate_brick_icon(w, h, d, color, brick_type, brick_icon_path)
        
        if os.path.exists(brick_icon_path):
            pdf.drawImage(ImageReader(brick_icon_path), x_pos, y_pos + (item_height - icon_size) // 2, 
                         width=icon_size, height=icon_size)
        text = f"{count} x {color_names.get(color, color)} ({brick_type})"
        pdf.drawString(x_pos + icon_size + 5, y_pos + (item_height - PDF_DESC_FONT_SIZE) // 2, text[:25])
        
        col += 1
        if col >= max_cols:
            col = 0
            y_pos -= item_height
            x_pos = list_x + 10
        else:
            x_pos += item_width

    pdf.showPage()

steps = []  # Глобальная переменная для доступа в render_step

def generate_pdf_instructions(cubes: List[Tuple[float, float, float, int, int, int, str, str]], 
                             output_file: str, progress_callback=None) -> None:
    if not cubes:
        logging.warning("No cubes for PDF generation")
        return

    global steps
    steps = get_layer_steps(cubes)
    total_steps = len(steps)
    logging.info(f"Generating PDF: {len(cubes)} cubes, {len(steps)} steps, path={output_file}")

    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
    color_names = {
        "#1A0000": "Very Dark Red",
        "#1A2F00": "Very Dark Green",
        "#001A2F": "Very Dark Blue",
        "#2F1A00": "Very Dark Orange",
        "#2F002F": "Very Dark Purple",
        "#1A1A00": "Very Dark Olive",
        "#002F2F": "Very Dark Teal",
        "#2F1A1A": "Very Dark Brown",
        "#1A2F1A": "Very Dark Forest",
        "#2F2F1A": "Very Dark Khaki"
    }

    full_model_buffer = io.BytesIO()
    render_full_model(cubes, full_model_buffer)

    try:
        pdf = canvas.Canvas(output_file, pagesize=PDF_PAGE_SIZE)
        add_parts_list_page(pdf, cubes, color_names)

        for step_index, current_layer in enumerate(steps):
            step_cubes = [cube for sublist in steps[:step_index + 1] for cube in sublist]
            z_level = current_layer[0][2]

            # Фон
            pdf.setFillColor(HexColor("#FFFFFF"))  # Белый фон
            pdf.rect(0, 0, PDF_PAGE_SIZE[0], PDF_PAGE_SIZE[1], fill=1)

            # Номер шага слева сверху
            pdf.setFillColorRGB(0, 0, 0)  # Черный текст
            pdf.setFont("Helvetica-Bold", PDF_HEADER_FONT_SIZE)
            pdf.drawString(20, PDF_HEADER_Y, f"Step {step_index + 1} of {total_steps}")

            # Картинка шага справа сверху в рамочке
            temp_step_path = os.path.join(TEMP_IMAGE_DIR, f"step_{step_index}.png")
            render_step(step_cubes, step_index, temp_step_path)
            if not os.path.exists(temp_step_path):
                logging.error(f"Rendering failed for step {step_index + 1}. Skipping image.")
                continue  # Пропускаем шаг, если изображение не создано
            if os.path.exists(temp_step_path):
                pdf.setStrokeColorRGB(0, 0, 0)
                pdf.setFillColor(HexColor("#FFFFFF"))
                pdf.roundRect(PDF_STEP_IMAGE_X - 5, PDF_STEP_IMAGE_Y - 5, 
                             PDF_STEP_IMAGE_WIDTH + 10, PDF_STEP_IMAGE_HEIGHT + 10, 10, stroke=1, fill=1)
                pdf.drawImage(ImageReader(temp_step_path), PDF_STEP_IMAGE_X, PDF_STEP_IMAGE_Y, 
                             width=PDF_STEP_IMAGE_WIDTH, height=PDF_STEP_IMAGE_HEIGHT)
                logging.info(f"Step {step_index + 1} image added to PDF: {temp_step_path}")
            else:
                logging.error(f"Step {step_index + 1} image not found at {temp_step_path}")

            # Список кубиков слева снизу в рамочке
            brick_counts = {}
            for cube in current_layer:
                x, y, z, w, h, d, color, brick_type = cube
                key = (w, h, d, color, brick_type)
                brick_counts[key] = brick_counts.get(key, 0) + 1

            # Рамка для списка
            list_x, list_y, list_width, list_height = 20, PDF_DESC_Y, 350, 450
            pdf.setStrokeColorRGB(0, 0, 0)
            pdf.setFillColor(HexColor("#FFFFFF"))
            pdf.roundRect(list_x, list_y, list_width, list_height, 10, stroke=1, fill=1)

            # Динамическое масштабирование
            num_items = len(brick_counts)
            max_cols = 2
            item_height = min(40, list_height // (num_items // max_cols + 1))
            item_width = list_width // max_cols
            icon_size = min(30, item_height - 10)

            y_pos = list_y + list_height - item_height
            x_pos = list_x + 10
            col = 0
            pdf.setFont("Helvetica", max(8, min(PDF_DESC_FONT_SIZE, item_height - 10)))  # Минимум 8pt
            pdf.setFillColorRGB(0, 0, 0)  # Черный текст

            for (w, h, d, color, brick_type), count in brick_counts.items():
                brick_icon_path = os.path.join(TEMP_IMAGE_DIR, f"brick_{w}_{h}_{d}_{color}_{brick_type}.png")
                generate_brick_icon(w, h, d, color, brick_type, brick_icon_path)
                
                if os.path.exists(brick_icon_path):
                    pdf.drawImage(ImageReader(brick_icon_path), x_pos, y_pos + (item_height - icon_size) // 2, 
                                 width=icon_size, height=icon_size)
                text = f"{count} x {color_names.get(color, color)} ({brick_type})"
                pdf.drawString(x_pos + icon_size + 5, y_pos + (item_height - PDF_DESC_FONT_SIZE) // 2, text[:20])
                
                col += 1
                if col >= max_cols:
                    col = 0
                    y_pos -= item_height
                    x_pos = list_x + 10
                else:
                    x_pos += item_width

            # Полная модель справа снизу
            pdf.setStrokeColorRGB(0, 0, 0)
            pdf.setFillColor(HexColor("#FFFFFF"))
            pdf.roundRect(PDF_FULL_MODEL_X - 5, PDF_FULL_MODEL_Y - 5, 
                         PDF_FULL_MODEL_SIZE + 10, PDF_FULL_MODEL_SIZE + 10, 10, stroke=1, fill=1)
            pdf.drawImage(ImageReader(full_model_buffer), PDF_FULL_MODEL_X, PDF_FULL_MODEL_Y, 
                         width=PDF_FULL_MODEL_SIZE, height=PDF_FULL_MODEL_SIZE)

            pdf.showPage()
            if progress_callback:
                progress_callback((step_index + 1) / total_steps)

        pdf.save()
        logging.info(f"PDF generated: {output_file}")
    except Exception as e:
        logging.error(f"PDF generation failed: {e}")
    finally:
        full_model_buffer.close()
        if os.path.exists(TEMP_IMAGE_DIR):
            shutil.rmtree(TEMP_IMAGE_DIR, ignore_errors=True)