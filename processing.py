import os
import logging
import numpy as np
import trimesh
from src.voxelization import adaptive_voxelization, voxel_grid_to_numpy
from src.brick_optimization import BrickPlacer, GreedyPlacementStrategy, SimulatedAnnealingPlacementStrategy, BranchAndBoundPlacementStrategy
from src.instruction_generation import generate_instructions, generate_pdf_instructions
from src.export import export_unique_bricks_stl, export_voxelized_stl
from src.config.config import SUPPORTED_EXTENSIONS

def process_model(model_path, scale_factor, max_depth, voxel_size, curvature_based, 
                 use_colors, method, allowed_sizes, output_dir, signals, 
                 clustering_method="connected", fill_hollow=True, minimal_support=False,
                 allow_top_layer=False, parallel_processing=False, render_steps=True,
                 do_generate_instructions=True, step_image_size=300):
    if signals._stopped:
        logging.debug("Process stopped before start")
        return
    try:
        signals.status.emit("Loading model")
        logging.info(f"Loading model from {model_path}")
        mesh = load_model(model_path)
        mesh.apply_scale(scale_factor)
        logging.info(f"Applied scale factor: {scale_factor}")
        signals.progress.emit(10)
        if signals._stopped:
            logging.debug("Stopped after loading model")
            return

        signals.status.emit("Filling model cavities")
        mesh.fill_holes()
        if not mesh.is_watertight:
            logging.warning("Model is not watertight, attempting to make it solid")
            try:
                mesh = trimesh.creation.extrude_polygon(mesh.outline(), height=mesh.extents[2])
            except Exception as e:
                logging.error(f"Failed to make model solid: {str(e)}")
                signals.error.emit("Failed to make model solid")
                return
        signals.progress.emit(20)
        if signals._stopped:
            logging.debug("Stopped after filling cavities")
            return

        signals.status.emit("Voxelizing with adaptive LEGO size")
        logging.info("Voxelizing model")
        voxel_grid = adaptive_voxelization(
            mesh, max_depth=max_depth, voxel_size=voxel_size, curvature_based=curvature_based
        )
        voxel_array = voxel_grid_to_numpy(voxel_grid)
        voxel_array = np.transpose(voxel_array, (2, 1, 0))  # z, y, x
        real_size = (voxel_array.shape[0] * voxel_grid.pitch, 
                     voxel_array.shape[1] * voxel_grid.pitch, 
                     voxel_array.shape[2] * voxel_grid.pitch)
        logging.info(f"Voxel grid real size (mm): {real_size}")
        signals.progress.emit(30)
        if signals._stopped:
            logging.debug("Stopped after voxelization")
            return

        signals.status.emit(f"Placing bricks (method={method})")
        logging.info(f"Placing bricks: method={method}")
        if method == "greedy":
            strategy = GreedyPlacementStrategy()
        elif method == "simulated_annealing":
            strategy = SimulatedAnnealingPlacementStrategy()
        elif method == "branch_and_bound":
            strategy = BranchAndBoundPlacementStrategy()
        else:
            raise ValueError(f"Unknown placement method: {method}")

        placer = BrickPlacer(strategy)
        def brick_progress(progress):
            if signals._stopped:
                logging.debug("Brick placement stopped")
                return True
            signals.progress.emit(int(30 + 30 * progress))
            return False
        cubes = placer.place_bricks(
                voxel_array, 
                use_colors=use_colors, 
                allowed_sizes=allowed_sizes, 
                fill_hollow=fill_hollow, 
                minimal_support=minimal_support, 
                progress_callback=brick_progress
            )
        logging.info(f"Brick placement completed: method={method}, cubes={len(cubes)}, colors used={use_colors}")
        signals.progress.emit(60)
        if signals._stopped:
            logging.debug("Stopped after brick placement")
            return

        if do_generate_instructions:
                signals.status.emit(f"Generating instructions (method={clustering_method})")
                logging.info("Generating instructions")
                instructions = generate_instructions(voxel_array, cubes, clustering_method, parallel_processing)
                signals.progress.emit(85)
        else:
                instructions = []
                signals.progress.emit(85)

        os.makedirs(output_dir, exist_ok=True)
        signals.status.emit("Exporting STL")
        stl_path = os.path.join(output_dir, "model.stl")
        export_voxelized_stl(voxel_array, stl_path)
        logging.info(f"STL exported to {stl_path}")
        signals.progress.emit(90)
        if signals._stopped:
            logging.debug("Stopped after STL export")
            return

        export_unique_bricks_stl(cubes, output_dir)
        signals.progress.emit(92)
        if signals._stopped:
            logging.debug("Stopped after unique bricks export")
            return

        if render_steps:  # Добавляем условие
                signals.status.emit("Rendering PDF steps")
                logging.info("Rendering PDF instructions")
                pdf_path = os.path.join(output_dir, "instructions.pdf")
                def pdf_progress(progress):
                    if signals._stopped:
                        logging.debug("PDF rendering stopped")
                        return True
                    signals.progress.emit(int(92 + 5 * progress))
                    return False
                generate_pdf_instructions(cubes, pdf_path, progress_callback=pdf_progress)
                logging.info(f"PDF exported: {pdf_path}")
                signals.progress.emit(95)
        else:
                pdf_path = None  # Если шаги не рендерятся, PDF не создается
                signals.progress.emit(95)
        if signals._stopped:
                logging.debug("Stopped after PDF rendering")
                return

        signals.status.emit("Finalizing PDF")
        signals.progress.emit(100)
        signals.finished.emit(cubes, instructions, pdf_path)

    except Exception as e:
        error_msg = f"Error during processing: {str(e)}"
        logging.error(error_msg)
        signals.error.emit(error_msg)
        signals.progress.emit(0)

def validate_file_path(file_path: str) -> None:
    if not isinstance(file_path, str):
        raise ValueError("Путь к файлу должен быть строкой")
    if not file_path.lower().endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(f"Поддерживаются только файлы {', '.join(SUPPORTED_EXTENSIONS)}")

def validate_mesh(mesh: trimesh.Trimesh) -> None:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Модель пуста или повреждена")
    if not mesh.is_watertight:
        logging.warning("Модель не герметична, возможны проблемы с вокселизацией")

def load_model(file_path: str, app=None) -> trimesh.Trimesh:
    logging.info(f"Loading model from {file_path}")
    validate_file_path(file_path)
    try:
        mesh = trimesh.load(file_path, force="mesh")
        bounds = mesh.bounds
        max_dim = max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1], bounds[1][2] - bounds[0][2])
        logging.info(f"Model bounds before scaling: {bounds}")
        logging.info(f"Max dimension before scaling: {max_dim} units")

        # Предполагаем, что если max_dim < 1, модель в метрах, масштабируем в миллиметры
        if max_dim < 1:  # Порог для определения единиц (1 метр = 1000 мм)
            scale_factor = 1000  # Метры -> миллиметры
            mesh.apply_scale(scale_factor)
            logging.info(f"Scaled model by {scale_factor} (assumed meters to millimeters)")
        
        bounds = mesh.bounds
        max_dim = max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1], bounds[1][2] - bounds[0][2])
        logging.info(f"Model bounds after scaling: {bounds}")
        logging.info(f"Max dimension after scaling: {max_dim} units")
        validate_mesh(mesh)
        logging.info(f"Model loaded: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
        if app:
            app.model_loaded = True
        return mesh
    except Exception as e:
        logging.error(f"Loading error: {e}")
        raise