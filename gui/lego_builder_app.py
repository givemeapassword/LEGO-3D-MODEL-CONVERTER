import os
import shutil
import sys
import time
import logging
import numpy as np
import pyvista as pv
import gc
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QDialogButtonBox,QDockWidget,QListWidgetItem,QLabel,QListWidget,QMessageBox,QApplication
from PyQt5.QtGui import QIcon, QPainterPath, QRegion, QPixmap,QImage
from PyQt5.QtCore import Qt, QSettings, QTimer, QRectF, QSize,QPropertyAnimation,QEasingCurve,QPoint
from src.config.config import (
    DEFAULT_OUTPUT_PATH, STUD_SIZE, TEMP_IMAGE_DIR, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, BASE_WIDTH, CAMERA_ANIMATION_STEPS, CAMERA_ANIMATION_INTERVAL,
    SNACKBAR_DISPLAY_DURATION, ROUNDING_RADIUS, CUBE_OPACITY, FLOOR_COLOR, FLOOR_EDGE_COLOR,
    FLOOR_OPACITY, LIGHT_DISTANCE_FACTOR, LIGHT_INTENSITY_TOP, LIGHT_INTENSITY_SIDES, LIGHT_INTENSITY_AMBIENT,
    BRICK_SIZES, PROGRESS_UPDATE_INTERVAL,SNACKBAR_ANIMATION_DURATION,THUMBNAIL_WIDTH,THUMBNAIL_ICON_SIZE
)
from src.gui.visualization import FLOOR_Z_POSITION, SceneRenderer, update_preview
from src.gui.model_interaction import set_view
from src.processing import load_model, process_model
from src.gui.gui_components import Header, ModelWindow, SettingsPanel, ActionButtons, ProgressLogs
from src.gui.gui_logger import QTextEditLogger
from src.gui.processing_thread import ProcessingThread
from src.signals import WorkerSignals
from PyQt5.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal


# Сигналы для рендеринга миниатюр
class ThumbnailWorkerSignals(QObject):
    finished = pyqtSignal(int, QIcon)  # Передаём индекс и готовую иконку
    error = pyqtSignal(str)

class ThumbnailWorker(QRunnable):
    def __init__(self, instructions, index):
        super().__init__()
        self.instructions = instructions
        self.index = index
        self.signals = ThumbnailWorkerSignals()

    def run(self):
        try:
            plotter = pv.Plotter(off_screen=True, window_size=(75, 75))
            plotter.set_background("#FFFFFF")
            renderer = SceneRenderer(plotter)
            # Рендерим кубики до текущего шага включительно
            renderer.render(self.instructions[:self.index + 1], 1.0, highlight_index=self.index)
            screenshot_path = f"temp_step_{self.index}.png"
            plotter.screenshot(screenshot_path)
            image = QImage(screenshot_path).scaled(75, 75, Qt.KeepAspectRatio, Qt.FastTransformation)
            icon = QIcon(QPixmap.fromImage(image))
            os.remove(screenshot_path)
            plotter.close()
            self.signals.finished.emit(self.index, icon)
        except Exception as e:
            self.signals.error.emit(f"Thumbnail {self.index} failed: {str(e)}")

class LegoBuilderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initialize_window()
        self.initialize_attributes()
        self.setup_ui()
        self.setup_logger()
        self.restore_layout()
        self.add_test_cube()
        self.load_settings()

    def initialize_window(self):
        self.setWindowTitle(WINDOW_TITLE)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setWindowIcon(QIcon("icons/logo_cube.png"))
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setObjectName("MainWindow")

    def initialize_attributes(self):
        self.base_width = BASE_WIDTH
        self.is_dark_theme = False
        self.dragging = False
        self.drag_position = None
        self.model_loaded = False
        self.pdf_path = None
        self.cubes = []
        self.model_path = None
        self.is_processing = False
        self.instructions = []
        self.original_mesh = None
        self.show_original = False
        self.start_time = None
        self.step_index = 0
        self.camera_animation_timer = QTimer(self)
        self.camera_animation_timer.timeout.connect(self.update_camera_animation)
        self.camera_animation_steps = CAMERA_ANIMATION_STEPS
        self.camera_animation_step = 0
        self.progress_history = []
        self.smoothing_factor = 0.2
        self.camera_start_pos = None
        self.camera_end_pos = None
        self.camera_start_focal = None
        self.camera_end_focal = None
        self.camera_start_up = None
        self.camera_end_up = None
        self.thumbnail_visible = False
        self.last_progress_update = 0
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_timer)
        self.elapsed_time = 0

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 20)
        main_layout.setSpacing(10)

        self.header = Header(self)
        self.header.mousePressEvent = self.header_mouse_press
        self.header.mouseMoveEvent = self.header_mouse_move
        self.header.mouseReleaseEvent = self.header_mouse_release
        main_layout.addWidget(self.header)

        self.model_window = ModelWindow(self)
        self.plotter = self.model_window.plotter

        settings_panel = SettingsPanel(self)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        right_layout.addWidget(settings_panel)

        content_layout = QHBoxLayout()
        content_layout.addSpacing(20)
        content_layout.addWidget(self.model_window)
        content_layout.addSpacing(20)
        content_layout.addWidget(right_panel)
        main_layout.addLayout(content_layout)

        custom_section = QWidget()
        bottom_layout = QHBoxLayout(custom_section)
        bottom_layout.setContentsMargins(20, 0, 0, 0)

        progress_container = QWidget()
        progress_container.setObjectName("progressContainer")
        progress_container_layout = QVBoxLayout(progress_container)
        progress_container_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_logs = ProgressLogs(self)
        progress_container_layout.addWidget(self.progress_logs)

        action_buttons = ActionButtons(self)

        bottom_layout.addWidget(progress_container)
        bottom_layout.addStretch()
        bottom_layout.addWidget(action_buttons)
        bottom_layout.addSpacing(20)
        main_layout.addWidget(custom_section)

        self.snackbar = QLabel(self.central_widget)
        self.snackbar.setObjectName("snackbar")
        self.snackbar.setVisible(False)
        self.snackbar_timer = QTimer(self)
        self.snackbar_timer.setSingleShot(True)
        self.snackbar_timer.timeout.connect(lambda: self.snackbar.setVisible(False))

        self.thumbnail_widget = QDockWidget("Steps", self)
        self.thumbnail_widget.setObjectName("thumbnailDock")
        self.thumbnail_widget.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.thumbnail_widget.setAllowedAreas(Qt.RightDockWidgetArea)
        self.thumbnail_widget.setFixedWidth(THUMBNAIL_WIDTH)
        self.addDockWidget(Qt.RightDockWidgetArea, self.thumbnail_widget, Qt.Horizontal)
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks)

        thumbnail_content = QWidget()
        thumbnail_layout = QVBoxLayout(thumbnail_content)
        thumbnail_layout.setContentsMargins(10, 10, 10, 10)
        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setObjectName("thumbnailList")
        self.thumbnail_list.itemClicked.connect(self.on_thumbnail_clicked)
        self.thumbnail_list.setIconSize(QSize(*THUMBNAIL_ICON_SIZE))
        thumbnail_layout.addWidget(self.thumbnail_list)
        self.thumbnail_widget.setWidget(thumbnail_content)
        self.thumbnail_widget.hide()

    def setup_logger(self):
        logger_handler = QTextEditLogger(self.log_display)
        logger_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(logger_handler)

    def on_thumbnail_clicked(self, item):
        index = self.thumbnail_list.row(item)
        self.step_index = index
        self.show_step_preview(index)

    def header_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPos() - self.pos()
            event.accept()
    
    def on_generate_clicked(self):
        logging.info(f"Generate clicked, model_loaded={self.model_loaded}")
        if not self.model_loaded:
            self.show_warning("Пожалуйста, сначала загрузите модель.")
            return
        # Логика генерации

    def on_pdf_clicked(self):
        logging.info(f"PDF clicked, model_loaded={self.model_loaded}")
        if not self.model_loaded:
            self.show_warning("Пожалуйста, сначала загрузите модель.")
            return
        # Логика экспорта в PDF

    def show_warning(self, message):
        logging.info(f"Attempting to show warning: {message}")
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle("Предупреждение")
        logging.info("Warning box created, executing...")
        QApplication.processEvents()  # Принудительная обработка событий
        msg_box.exec_()
        logging.info("Warning box executed")

    def header_mouse_move(self, event):
        if self.dragging:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def header_mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            event.accept()

    def animate_camera(self, start_pos, end_pos, start_focal, end_focal, start_up, end_up):
        self.camera_start_pos = np.array(start_pos)
        self.camera_end_pos = np.array(end_pos)
        self.camera_start_focal = np.array(start_focal)
        self.camera_end_focal = np.array(end_focal)
        self.camera_start_up = np.array(start_up)
        self.camera_end_up = np.array(end_up)
        self.camera_animation_step = 0
        self.camera_animation_timer.start(CAMERA_ANIMATION_INTERVAL)

    def update_camera_animation(self):
        if self.camera_animation_step >= self.camera_animation_steps:
            self.camera_animation_timer.stop()
            self.plotter.camera_position = (self.camera_end_pos, self.camera_end_focal, self.camera_end_up)
            self.plotter.update()
            return
        t = self.camera_animation_step / self.camera_animation_steps
        t = t * t * (3 - 2 * t)
        current_pos = (1 - t) * self.camera_start_pos + t * self.camera_end_pos
        current_focal = (1 - t) * self.camera_start_focal + t * self.camera_end_focal
        current_up = (1 - t) * self.camera_start_up + t * self.camera_end_up
        self.plotter.camera_position = (current_pos, current_focal, current_up)
        self.plotter.update()
        self.camera_animation_step += 1

    def update_voxel_size(self):
        value = self.voxel_size.value() / 10.0
        self.voxel_size_max.setText(f"{value:.1f}")

    def closeEvent(self, event):
        logging.info("Application closing, terminating all processes")
        self.save_settings()
        self.save_layout()
        if hasattr(self, 'processing_thread') and self.processing_thread.isRunning():
            self.worker_signals._stopped = True
            self.processing_thread.stop()  # Останавливаем поток
            logging.info("Processing thread stopped")
        self.is_processing = False
        self.is_generating = False
        self.progress_timer.stop()
        # Очистка временных файлов
        if os.path.exists(TEMP_IMAGE_DIR):
            shutil.rmtree(TEMP_IMAGE_DIR, ignore_errors=True)
        logging.info("All processes terminated, resources cleaned up")
        event.accept()  # Закрываем приложение

    def toggle_thumbnail(self):
        if self.thumbnail_visible:
            self.thumbnail_widget.hide()
            self.resize(self.base_width, self.height())
        else:
            self.thumbnail_widget.show()
            new_width = self.base_width + THUMBNAIL_WIDTH
            self.resize(new_width, self.height())
        self.thumbnail_visible = not self.thumbnail_visible

    def resizeEvent(self, event):
        path = QPainterPath()
        rect = self.rect()
        path.addRoundedRect(QRectF(rect), ROUNDING_RADIUS, ROUNDING_RADIUS)
        region = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)
        super().resizeEvent(event)

    def on_processing_error(self, msg):
        self.show_snackbar(f"Error: {msg}")
        self.is_processing = False
        self.is_generating = False
        self.generate_button.setText("Generation")
        self.generate_button.setIcon(QIcon("icons/generate_cube.png"))
        self.generate_button.setStyleSheet("""
            #generateButton {
                font-family: 'Poppins';
                background-color: #f05d22;
                color: #ffffff;
                font-size: 24px;
                border-radius: 20px;
            }
            #generateButton:hover {
                background-color: #d94a1a;
            }
        """)
        self.progress_timer.stop()

    def restore_layout(self):
        settings = QSettings("LegoBuilder", "AppLayout")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        window_state = settings.value("windowState")
        if window_state:
            self.restoreState(window_state)

    def save_layout(self):
        settings = QSettings("LegoBuilder", "AppLayout")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def add_test_cube(self):
        cube = pv.Cube(center=(0, 0, 0), x_length=20, y_length=20, z_length=20)
        self.plotter.add_mesh(cube, color="#FF0000", show_edges=False, opacity=CUBE_OPACITY, lighting=True)
        max_size = 200
        floor = pv.Plane(center=(0, 0, FLOOR_Z_POSITION), i_size=max_size, j_size=max_size)
        self.plotter.add_mesh(floor, color=FLOOR_COLOR, show_edges=True, edge_color=FLOOR_EDGE_COLOR,
                              opacity=FLOOR_OPACITY, lighting=False)
        self.plotter.remove_all_lights()
        light_distance = max_size * LIGHT_DISTANCE_FACTOR
        light_positions = [(0, 0, light_distance), (0, light_distance, 0), (0, -light_distance, 0),
                           (-light_distance, 0, 0), (light_distance, 0, 0)]
        intensities = [LIGHT_INTENSITY_TOP] + [LIGHT_INTENSITY_SIDES] * 4
        for pos, intensity in zip(light_positions, intensities):
            self.plotter.add_light(pv.Light(position=pos, color="white", intensity=intensity,
                                            positional=True, show_actor=False))
        self.plotter.add_light(pv.Light(color="white", intensity=LIGHT_INTENSITY_AMBIENT,
                                        positional=False, show_actor=False))
        self.plotter.reset_camera()
        logging.info("Test cube added to scene.")

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_path.setText(folder)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open STL/OBJ File", "", "3D Models (*.stl *.obj)")
        if file_path:
            self.model_path = file_path
            logging.info(f"Loading model from: {file_path}")
            # Очищаем текущую сцену
            self.cubes = []
            self.instructions = []
            self.original_mesh = None
            self.plotter.clear()  # Полная очистка plotter
            try:
                self.original_mesh = load_model(self.model_path, self)  # Передаем self
                self.show_snackbar(f"Loaded: {file_path}")
                # self.model_loaded = True  # Убираем, так как обновляется в load_model
                self.show_original = True
                update_preview(self)
                self.model_window.view_model_button.setChecked(True)
                self.model_window.view_voxel_button.setChecked(False)
                self.plotter.reset_camera()  # Сброс камеры после загрузки
                self.generate_button.setEnabled(True)
            except Exception as e:
                self.show_snackbar(f"Failed to load model: {str(e)}")
                logging.error(f"Model loading failed: {str(e)}")

    def open_pdf(self):
        if not self.pdf_path or not os.path.exists(self.pdf_path):
            self.show_snackbar("Please generate the model first")
            return
        try:
            if sys.platform == "win32":
                os.startfile(self.pdf_path)
            else:
                import subprocess
                subprocess.run(["xdg-open", self.pdf_path])
        except Exception as e:
            self.show_snackbar(f"Failed to open PDF: {str(e)}")
            logging.error(f"Failed to open PDF: {str(e)}")

    def show_snackbar(self, message, duration=SNACKBAR_DISPLAY_DURATION):
        logging.info(f"Showing snackbar: {message}")
        self.snackbar.setText(message)
        self.snackbar.setGeometry(20, self.height()-100, 350, 80)  # Фиксированный размер
        self.snackbar.setStyleSheet("background-color: #f05d22; color: #ffffff;")  # Явный стиль
        self.snackbar.show()
        self.snackbar.raise_()  # Поднимаем над другими виджетами
        QApplication.processEvents()  # Принудительный рендеринг
        logging.info("Snackbar shown")
        self.snackbar_timer.timeout.disconnect()
        self.snackbar_timer.timeout.connect(lambda: self.snackbar.hide())
        self.snackbar_timer.start(duration)

    def hide_snackbar(self):
        snackbar_width = self.snackbar.width()
        window_width = self.width()
        x_pos = (window_width - snackbar_width) // 2
        y_end = self.height()

        anim_out = QPropertyAnimation(self.snackbar, b"pos")
        anim_out.setDuration(SNACKBAR_ANIMATION_DURATION)
        anim_out.setStartValue(self.snackbar.pos())
        anim_out.setEndValue(QPoint(x_pos, y_end))
        anim_out.setEasingCurve(QEasingCurve.InOutCubic)
        anim_out.finished.connect(lambda: self.snackbar.setVisible(False))
        anim_out.start()
        logging.info("Snackbar hidden")

    def generate(self):
        if not self.model_path:
            self.show_snackbar("Пожалуйста, сначала загрузите модель!")
            return
        if self.is_processing:
            return
        self.is_processing = True
        self.is_generating = False
        self.start_time = time.time()
        self.elapsed_time = 0
        self.progress.setValue(0)
        self.worker_signals = WorkerSignals()
        self.worker_signals._stopped = False
        self.generate_button.setEnabled(False)

        model_path = self.model_path
        scale_factor = self.scale_factor.value() / 100.0  # Из слайдера Scale Factor
        output_dir = self.output_path.text()
        max_depth = self.max_depth_slider.value()
        voxel_size_map = {
            "1 stud (High Detail)": STUD_SIZE,
            "2 studs (Medium Detail)": STUD_SIZE * 2,
            "3 studs (Low Detail)": STUD_SIZE * 3
        }
        voxel_size = voxel_size_map[self.voxel_size.currentText()]
        curvature_based = self.curvature_based.isChecked()
        use_colors = self.use_colors.isChecked()

        selected_items = self.brick_sizes.selectedItems()
        if selected_items:
            selected_sizes = []
            for item in selected_items:
                text = item.text()
                size_part, type_part = text.split(" (")
                w, h, d = map(int, size_part.split("x"))
                brick_type = type_part[:-1]
                selected_sizes.append((w, h, d, brick_type))
            allowed_sizes = selected_sizes
        else:
            allowed_sizes = BRICK_SIZES

        placement_method_map = {
            "Greedy (Fast)": "greedy",
            "Simulated Annealing": "simulated_annealing",
            "Branch and Bound": "branch_and_bound"
        }
        placement_method = placement_method_map[self.placement_method.currentText()]
        clustering_method = "connected" if self.instruction_style.currentText() == "Fast Grouping" else "dbscan"

        # Обработка Fill Mode
        fill_mode = self.fill_mode.currentText()
        fill_hollow = (fill_mode == "Full Fill")
        minimal_support = (fill_mode == "Minimal Supports")

        # Новые параметры
        allow_top_layer = self.allow_top_layer.isChecked()
        parallel_processing = self.parallel_processing.isChecked()
        render_steps = self.render_steps.isChecked()
        generate_instructions = self.generate_instructions.isChecked()
        step_image_size = self.step_image_size.value()

        logging.debug(f"Starting generation with: voxel_size={voxel_size}, scale_factor={scale_factor}, "
                    f"allowed_sizes={allowed_sizes}, placement_method={placement_method}, "
                    f"clustering_method={clustering_method}, fill_hollow={fill_hollow}, "
                    f"minimal_support={minimal_support}, allow_top_layer={allow_top_layer}, "
                    f"parallel_processing={parallel_processing}, render_steps={render_steps}, "
                    f"generate_instructions={generate_instructions}")
        logging.info("Starting model generation")
        self.progress_history.clear()

        self.processing_thread = ProcessingThread(
            self,
            process_model,
            self.worker_signals,
            model_path, scale_factor, max_depth, voxel_size,
            curvature_based, use_colors, placement_method,
            allowed_sizes, output_dir, self.worker_signals,
            clustering_method=clustering_method,
            fill_hollow=fill_hollow, minimal_support=minimal_support,
            allow_top_layer=allow_top_layer, parallel_processing=parallel_processing,
            render_steps=render_steps, do_generate_instructions=generate_instructions,
            step_image_size=step_image_size
        )
        self.is_generating = True
        self.worker_signals.progress.connect(self.update_progress)
        self.worker_signals.status.connect(self.progress_stage.setText)
        self.worker_signals.status.connect(lambda status: logging.info(status))
        self.worker_signals.finished.connect(self.on_processing_finished)

        def handle_error(msg):
            self.show_snackbar(f"Ошибка: {msg}")
            self.is_processing = False
            self.generate_button.setEnabled(True)

        self.worker_signals.error.connect(handle_error)
        self.worker_signals.finished.connect(lambda: self.generate_button.setEnabled(True))
        self.processing_thread.start()
        self.generate_button.setEnabled(False)

    def update_progress(self, value):
        current_time = time.time()
        if current_time - self.last_progress_update >= PROGRESS_UPDATE_INTERVAL or value == 100:
            self.progress.setValue(value)
            if value == 0:
                self.elapsed_time = 0
                self.progress_timer.stop()
            elif value > 0 and not self.progress_timer.isActive():
                self.progress_timer.start(1000)  # Обновление каждую секунду
            self.progress_label.setText(f"{value}% ({self.elapsed_time}s elapsed)")
        self.last_progress_update = current_time

    def update_progress_timer(self):
        self.elapsed_time += 1
        value = self.progress.value()
        self.progress_label.setText(f"{value}% ({self.elapsed_time}s elapsed)")

    def calculate_complexity(self, cubes, instructions, elapsed):
        # 1. Количество кубиков (вес: 30%)
        cube_count = len(cubes)
        cube_score = min(cube_count / 1000, 1.0) * 30  # Нормализация до 1000 кубиков

        # 2. Разнообразие типов кирпичей (вес: 20%)
        brick_types = set((cube[0], cube[1], cube[2], cube[3]) for cube in cubes)  # Предполагаем, что cube = (w, h, d, type)
        type_score = min(len(brick_types) / len(BRICK_SIZES), 1.0) * 20  # Нормализация по количеству доступных типов

        # 3. Количество шагов инструкции (вес: 20%)
        step_count = len(instructions)
        step_score = min(step_count / 50, 1.0) * 20  # Нормализация до 50 шагов

        # 4. Плотность соединений (вес: 20%)
        # Примерная оценка: считаем среднее количество кубиков на шаг
        if step_count > 0:
            density = cube_count / step_count
            density_score = min(density / 20, 1.0) * 20  # Нормализация до 20 кубиков на шаг
        else:
            density_score = 0

        # 5. Время генерации (вес: 10%)
        time_score = min(elapsed / 60, 1.0) * 10  # Нормализация до 60 секунд

        # Итоговый балл (0-100)
        total_score = cube_score + type_score + step_score + density_score + time_score

        # Преобразование в категорию
        if total_score >= 75:
            complexity = "Extreme"
        elif total_score >= 50:
            complexity = "High"
        elif total_score >= 25:
            complexity = "Medium"
        else:
            complexity = "Low"

        logging.info(f"Complexity calculated: total_score={total_score:.1f}, "
                    f"cubes={cube_count}, types={len(brick_types)}, steps={step_count}, "
                    f"density={density_score:.1f}, time={elapsed:.1f}s")
        return complexity, total_score

    def on_processing_finished(self, cubes, instructions, pdf_path):
        self.is_processing = False
        self.progress_timer.stop()
        self.model_window.view_model_button.setChecked(False)
        self.model_window.view_voxel_button.setChecked(True)
        if not self.worker_signals._stopped:
            self.cubes = cubes
            self.instructions = instructions
            elapsed = time.time() - self.start_time
            complexity, total_score = self.calculate_complexity(cubes, instructions, elapsed)
            self.pdf_path = pdf_path
            self.show_snackbar(f"Model generated! PDF saved at: {self.pdf_path}")
            logging.info(f"Generation completed: cubes={len(cubes)}, steps={len(instructions)}, "
                        f"time={int(elapsed)}s, complexity={complexity} ({total_score:.1f})")
            self.bricks_value.setText(str(len(cubes)))
            self.time_value.setText(f"{int(elapsed)}s")
            self.complexity_value.setText(f"{complexity} ({int(total_score)}%)")
            self.show_original = False
            update_preview(self)

            # Очистка списка миниатюр
            self.thumbnail_list.clear()
            self.thumbnail_visible = False

            # Создаём пул потоков
            self.thread_pool = QThreadPool.globalInstance()
            self.thread_pool.setMaxThreadCount(2)

            # Рендерим первые 10 миниатюр асинхронно
            shared_plotter = pv.Plotter(off_screen=True, window_size=(75, 75))
            shared_plotter.set_background("#FFFFFF")
            initial_batch = min(5, len(self.instructions))
            for i in range(initial_batch):
                    worker = ThumbnailWorker(self.instructions, i)
                    worker.signals.finished.connect(self.add_thumbnail)
                    worker.signals.error.connect(lambda msg: logging.error(msg))
                    self.thread_pool.start(worker)

            # Оставшиеся миниатюры рендерим постепенно
            if len(self.instructions) > initial_batch:
                self.remaining_thumbnails = list(range(initial_batch, len(self.instructions)))
                QTimer.singleShot(200, self.process_next_thumbnail)
                QTimer.singleShot(200, shared_plotter.close)

            gc.collect()
        self.generate_button.setText("Generation")
        self.generate_button.setIcon(QIcon("icons/generate_cube.png"))
        self.generate_button.setStyleSheet("""
            #generateButton {
                font-family: 'Poppins';
                background-color: #f05d22;
                color: #ffffff;
                font-size: 24px;
                border-radius: 20px;
            }
            #generateButton:hover {
                background-color: #d94a1a;
            }
        """)
        self.generate_button.setEnabled(True)

    def show_step_preview(self, index):
        if index >= len(self.instructions):
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Step {index + 1} Preview")
        dialog.setMinimumSize(800, 600)

        scene = QGraphicsScene()
        view = QGraphicsView(scene, dialog)

        temp_plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        renderer = SceneRenderer(temp_plotter)
        # Рендерим только кубики до текущего шага включительно
        renderer.render(self.instructions[:index + 1], 1.0, highlight_index=index)
        screenshot_path = f"temp_preview_step_{index}.png"
        temp_plotter.screenshot(screenshot_path)

        pixmap = QPixmap(screenshot_path)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        view.fitInView(item, Qt.KeepAspectRatio)
        view.scale(20,20)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close, dialog)
        buttons.accepted.connect(lambda: self.save_preview(screenshot_path, index))
        buttons.rejected.connect(dialog.reject)

        layout = QVBoxLayout(dialog)
        layout.addWidget(view)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        logging.info(f"Opened step {index + 1}")
        dialog.exec_()
        os.remove(screenshot_path)
        temp_plotter.close()

    def save_preview(self, screenshot_path, index):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Preview", f"step_{index + 1}.png", "PNG Images (*.png)")
        if save_path:
            import shutil
            shutil.copy(screenshot_path, save_path)
            self.show_snackbar(f"Preview saved!")
            logging.info(f"Saved step {index + 1} to {save_path}")

    def set_view(self, view, camera_pos=None, camera_focal_point=None, camera_view_up=None):
        set_view(self, view, camera_pos, camera_focal_point, camera_view_up)
    
    def toggle_generation(self):
        if not self.is_processing:
            self.generate()
            self.generate_button.setText("Stop")
            self.generate_button.setIcon(QIcon("icons/stop.png"))  # Добавьте иконку stop
            self.generate_button.setStyleSheet("""
                #generateButton {
                    font-family: 'Poppins';
                    background-color: #FF4040;
                    color: #ffffff;
                    font-size: 24px;
                    border-radius: 20px;
                }
                #generateButton:hover {
                    background-color: #d32f2f;
                }
            """)
        elif hasattr(self, 'is_generating') and self.is_generating:
            self.stop_generation()

    def stop_generation(self):
        if hasattr(self, 'processing_thread') and self.processing_thread.isRunning():
            self.worker_signals._stopped = True  # Устанавливаем флаг
            self.processing_thread.stop()  # Прерываем поток
        self.is_processing = False
        self.is_generating = False
        self.progress.setValue(0)
        self.progress_timer.stop()
        # Обновляем кнопку
        self.generate_button.setText("Generation")
        self.generate_button.setIcon(QIcon("icons/generate_cube.png"))
        self.generate_button.setStyleSheet("""
            #generateButton {
                font-family: 'Poppins';
                background-color: #f05d22;
                color: #ffffff;
                font-size: 24px;
                border-radius: 20px;
            }
            #generateButton:hover {
                background-color: #d94a1a;
            }
        """)
        self.generate_button.setEnabled(True)  # Включаем кнопку
        self.show_snackbar("Generation stopped")

            # Метод для добавления миниатюры в список
    def add_thumbnail(self, index, icon):
        item = QListWidgetItem(icon, f"Step {index + 1}")
        self.thumbnail_list.addItem(item)

    # Метод для постепенной обработки оставшихся миниатюр
    def process_next_thumbnail(self):
        if not hasattr(self, 'remaining_thumbnails') or not self.remaining_thumbnails:
            return
        index = self.remaining_thumbnails.pop(0)
        worker = ThumbnailWorker(self.instructions, index)
        worker.signals.finished.connect(self.add_thumbnail)
        worker.signals.error.connect(lambda msg: logging.error(msg))
        self.thread_pool.start(worker)
        if self.remaining_thumbnails:  # Если остались ещё миниатюры
            QTimer.singleShot(50, self.process_next_thumbnail)  # Очередной через 50 мс

    def load_settings(self):
        self.settings = QSettings("LegoBuilderPro", "Settings")
        # Вкладка Settings
        self.fill_mode.setCurrentText(self.settings.value("fill_mode", "Full Fill"))
        self.voxel_size.setCurrentText(self.settings.value("voxel_size", "1 stud (High Detail)"))
        self.max_depth_slider.setValue(self.settings.value("max_depth", 10, type=int))
        self.curvature_based.setChecked(self.settings.value("curvature_based", False, type=bool))
        self.minimal_support.setChecked(self.settings.value("minimal_support", False, type=bool))
        self.scale_factor.setValue(self.settings.value("scale_factor", 100, type=int))
        self.instruction_style.setCurrentText(self.settings.value("instruction_style", "Fast Grouping"))
        self.placement_method.setCurrentText(self.settings.value("placement_method", "Greedy (Fast)"))
        self.use_colors.setChecked(self.settings.value("use_colors", True, type=bool))
        self.render_steps.setChecked(self.settings.value("render_steps", True, type=bool))
        self.generate_instructions.setChecked(self.settings.value("generate_instructions", True, type=bool))
        self.parallel_processing.setChecked(self.settings.value("parallel_processing", False, type=bool))
        self.output_path.setText(self.settings.value("output_path", DEFAULT_OUTPUT_PATH))
        self.export_voxelized.setChecked(self.settings.value("export_voxelized", True, type=bool))
        self.export_unique_bricks.setChecked(self.settings.value("export_unique_bricks", True, type=bool))
        self.step_image_size.setValue(self.settings.value("step_image_size", 300, type=int))
        self.log_level.setCurrentText(self.settings.value("log_level", "INFO"))
        self.allow_top_layer.setChecked(self.settings.value("allow_top_layer", False, type=bool))
        # Вкладка Cubs
        selected_bricks = self.settings.value("brick_sizes", [f"{w}x{h}x{d} ({t})" for w, h, d, t in BRICK_SIZES], type=list)
        for item in [self.brick_sizes.item(i) for i in range(self.brick_sizes.count())]:
            item.setSelected(item.text() in selected_bricks)

    def save_settings(self):
        self.settings = QSettings("LegoBuilderPro", "Settings")
        # Вкладка Settings
        self.settings.setValue("fill_mode", self.fill_mode.currentText())
        self.settings.setValue("voxel_size", self.voxel_size.currentText())
        self.settings.setValue("max_depth", self.max_depth_slider.value())
        self.settings.setValue("curvature_based", self.curvature_based.isChecked())
        self.settings.setValue("minimal_support", self.minimal_support.isChecked())
        self.settings.setValue("scale_factor", self.scale_factor.value())
        self.settings.setValue("instruction_style", self.instruction_style.currentText())
        self.settings.setValue("placement_method", self.placement_method.currentText())
        self.settings.setValue("use_colors", self.use_colors.isChecked())
        self.settings.setValue("render_steps", self.render_steps.isChecked())
        self.settings.setValue("generate_instructions", self.generate_instructions.isChecked())
        self.settings.setValue("parallel_processing", self.parallel_processing.isChecked())
        self.settings.setValue("output_path", self.output_path.text())
        self.settings.setValue("export_voxelized", self.export_voxelized.isChecked())
        self.settings.setValue("export_unique_bricks", self.export_unique_bricks.isChecked())
        self.settings.setValue("step_image_size", self.step_image_size.value())
        self.settings.setValue("log_level", self.log_level.currentText())
        self.settings.setValue("allow_top_layer", self.allow_top_layer.isChecked())
        # Вкладка Cubs
        selected_bricks = [item.text() for item in self.brick_sizes.selectedItems()]
        self.settings.setValue("brick_sizes", selected_bricks if selected_bricks else [f"{w}x{h}x{d} ({t})" for w, h, d, t in BRICK_SIZES])

    def view_model(self):
            if not self.original_mesh:
                self.show_snackbar("No original model loaded!")
                return
            logging.info("Switching to original model view")
            self.show_original = True
            self.model_window.view_model_button.setChecked(True)  # Устанавливаем активное состояние
            self.model_window.view_voxel_button.setChecked(False)  # Снимаем активное состояние
            self.camera_animation_timer.stop()
            self.plotter.clear()
            update_preview(self)
            self.plotter.reset_camera()

    def view_voxel(self):
        if not self.cubes:
            self.show_snackbar("No voxel assembly generated!")
            return
        logging.info("Switching to voxel assembly view")
        self.show_original = False
        self.model_window.view_model_button.setChecked(False)  # Снимаем активное состояние
        self.model_window.view_voxel_button.setChecked(True)  # Устанавливаем активное состояние
        self.plotter.clear()
        update_preview(self)
        self.plotter.reset_camera()
        