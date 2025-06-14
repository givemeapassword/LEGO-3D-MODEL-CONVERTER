import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QComboBox, QLineEdit, QProgressBar, QListWidget, QListWidgetItem, QCheckBox,
    QTextEdit, QTabWidget, QGraphicsDropShadowEffect, QFormLayout, QSizePolicy,QScrollArea,QSlider
)
from PyQt5.QtGui import QIcon, QPixmap, QColor,QPainterPath,QRegion
from PyQt5.QtCore import Qt, QSize,QRectF
from src.config.config import (
    HEADER_HEIGHT, SHADOW_BLUR_RADIUS, SHADOW_X_OFFSET, SHADOW_Y_OFFSET, SHADOW_ALPHA,
    LOGO_SIZE, BUTTON_GROUP_SIZE, MODEL_WINDOW_MIN_WIDTH, SETTINGS_PANEL_MIN_WIDTH,
    SETTINGS_PANEL_MIN_HEIGHT, ACTION_BUTTON_SIZE, SMALL_BUTTON_SIZE, ICON_SIZE, 
    OUTPUT_PATH_BUTTON_SIZE, TOGGLE_BUTTON_SIZE,
    PROGRESS_HEIGHT, DEFAULT_OUTPUT_PATH, BRICK_SIZES
)
from src.gui.view_cube import ViewCube
from pyvistaqt import QtInteractor

class Header(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.setObjectName("header")
        self.setFixedHeight(HEADER_HEIGHT)
        self.apply_shadow()
        self.setup_layout(parent)

    def apply_shadow(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(SHADOW_BLUR_RADIUS)
        shadow.setXOffset(SHADOW_X_OFFSET)
        shadow.setYOffset(SHADOW_Y_OFFSET)
        shadow.setColor(QColor(12, 12, 13, SHADOW_ALPHA))
        self.setGraphicsEffect(shadow)

    def setup_layout(self, parent):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(25, 0, 0, 0)
        layout.setSpacing(0)

        logo = QLabel()
        pixmap = QPixmap("icons/logo_cube.png")
        logo.setPixmap(pixmap.scaled(*LOGO_SIZE, Qt.KeepAspectRatio))
        layout.addWidget(logo)

        title = QLabel("Lego Builder Pro")
        title.setObjectName("titleLabel")  # Добавляем objectName
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title, stretch=1)

        button_group = QWidget()
        button_group_layout = QHBoxLayout(button_group)
        button_group_layout.setContentsMargins(0, 0, 0, 0)
        button_group_layout.setSpacing(0)

        minimize_button = QPushButton("−")
        minimize_button.setObjectName("minimizeButton")
        minimize_button.setFixedSize(*BUTTON_GROUP_SIZE)
        minimize_button.clicked.connect(parent.showMinimized)
        minimize_button.setToolTip("Свернуть приложение")

        close_button = QPushButton("x")
        close_button.setObjectName("closeButton")
        close_button.setFixedSize(*BUTTON_GROUP_SIZE)
        close_button.clicked.connect(parent.close)
        close_button.setToolTip("Закрыть приложение")

        button_group_layout.addWidget(minimize_button)
        button_group_layout.addWidget(close_button)
        layout.addWidget(button_group)

class ModelWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.setObjectName("modelContainer")
        layout = QVBoxLayout(self)
        frame = QWidget(self)
        frame.setStyleSheet("background-color: #C0C0C0; border-radius: 20px;")
        frame_layout = QVBoxLayout(frame)
        layout.addWidget(frame)

        model_window = QWidget()
        model_window.setObjectName("modelWindow")
        model_window.setMinimumSize(MODEL_WINDOW_MIN_WIDTH, 0)
        model_window.setStyleSheet("background-color: #C0C0C0;")
        model_layout = QVBoxLayout(model_window)
        model_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(model_window)
        self.plotter.set_background("#C0C0C0")
        model_layout.addWidget(self.plotter)

        frame_layout.addWidget(model_window)

        # Оверлей для ViewCube (в правом верхнем углу)
        overlay_widget = QWidget(self.plotter)
        overlay_widget.setAttribute(Qt.WA_TranslucentBackground, True)
        overlay_layout = QHBoxLayout(overlay_widget)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setAlignment(Qt.AlignCenter)
        view_cube = ViewCube(overlay_widget, lambda view, camera_pos=None, camera_focal_point=None, camera_view_up=None:
                             parent.set_view(view, camera_pos, camera_focal_point, camera_view_up))
        overlay_layout.addWidget(view_cube)
        overlay_widget.setGeometry(0, 0, 150, 150)

        # Отдельный оверлей для кнопок (в левом нижнем углу)
        self.buttons_overlay = QWidget(self)  # Сохраняем как атрибут класса
        self.buttons_overlay.setObjectName("buttonsOverlay")

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(2, 2)
        self.buttons_overlay.setGraphicsEffect(shadow)
        buttons_layout = QHBoxLayout(self.buttons_overlay)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Кнопка для переключения на вид модели
        self.view_model_button = QPushButton(self.buttons_overlay)
        self.view_model_button.setObjectName("viewModelButton")
        self.view_model_button.setCheckable(True)
        self.view_model_button.setChecked(True)
        self.view_model_button.clicked.connect(parent.view_model)
        self.view_model_button.setToolTip("Switch to original model view")
        self.view_model_button.setFocusPolicy(Qt.StrongFocus)
        self.view_model_button.setMouseTracking(True)

        # Кнопка для переключения на воксельную сборку
        self.view_voxel_button = QPushButton(self.buttons_overlay)
        self.view_voxel_button.setObjectName("viewVoxelButton")
        self.view_voxel_button.setCheckable(True)
        self.view_voxel_button.clicked.connect(parent.view_voxel)
        self.view_voxel_button.setToolTip("Switch to voxel assembly view")
        self.view_voxel_button.setFocusPolicy(Qt.StrongFocus)
        self.view_voxel_button.setMouseTracking(True)

        buttons_layout.addWidget(self.view_model_button)
        buttons_layout.addWidget(self.view_voxel_button)

        # Устанавливаем размер и позицию buttons_overlay (183x80px)
        self.buttons_overlay.setGeometry(20, self.height() - 100, 183, 80)

        # Обновляем позицию и маску при изменении размера окна
        self.resizeEvent = self._resizeEvent

    def _resizeEvent(self, event):
        # Обновляем позицию
            self.buttons_overlay.setGeometry(20, self.height() - 100, 183, 80)

            # Применяем маску с закруглёнными углами
            path = QPainterPath()
            rect = self.buttons_overlay.rect()
            path.addRoundedRect(QRectF(rect), 13, 13)  # Радиус скругления 12px, как в CSS
            region = QRegion(path.toFillPolygon().toPolygon())
            self.buttons_overlay.setMask(region)

            super().resizeEvent(event)

class SettingsPanel(QTabWidget):
    def __init__(self, parent):
        super().__init__()
        self.setObjectName("settingsPanel")
        self.setMinimumSize(SETTINGS_PANEL_MIN_WIDTH, SETTINGS_PANEL_MIN_HEIGHT)
        self.apply_shadow()
        self.setup_tabs(parent)

    def apply_shadow(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(SHADOW_BLUR_RADIUS)
        shadow.setXOffset(SHADOW_X_OFFSET)
        shadow.setYOffset(SHADOW_Y_OFFSET)
        shadow.setColor(QColor(12, 12, 13, SHADOW_ALPHA))
        self.setGraphicsEffect(shadow)

    def setup_tabs(self, parent):
        # Вкладка Settings
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.verticalScrollBar().valueChanged.connect(settings_scroll.viewport().update)
        settings_scroll.viewport().setAttribute(Qt.WA_OpaquePaintEvent)
        settings_scroll.viewport().setAutoFillBackground(True)
        settings_widget = QWidget()
        settings_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        settings_widget.setMinimumHeight(0)
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(20, 20, 20, 20)
        settings_layout.setSpacing(10)

        # Fill Mode
        parent.fill_mode = QComboBox()
        parent.fill_mode.addItems(["Full Fill", "Minimal Supports", "No Fill"])
        parent.fill_mode.setCurrentIndex(0)
        parent.fill_mode.setToolTip("Полное заполнение: заполняет всю модель. Минимальные опоры: только под верхним слоем. Без заполнения: полая модель с боковой поддержкой.")
        settings_layout.addWidget(QLabel("Fill Mode"))
        settings_layout.addWidget(parent.fill_mode)

        # Voxel Size
        voxel_size_label = QLabel("Voxel Size (mm)")
        voxel_size_label.setToolTip("Устанавливает размер вокселя для совместимости с LEGO")
        parent.voxel_size = QComboBox()
        parent.voxel_size.addItems(["1 stud (High Detail)", "2 studs (Medium Detail)", "3 studs (Low Detail)"])
        parent.voxel_size.setCurrentIndex(0)
        parent.voxel_size.setToolTip("3.2 мм: высокая детализация (плиты), 6.4 мм: сбалансировано, 9.6 мм: низкая детализация (кирпичи)")
        settings_layout.addWidget(voxel_size_label)
        settings_layout.addWidget(parent.voxel_size)

        # Voxelization Depth (Advanced)
        max_depth_label = QLabel("Voxelization Depth (Advanced)")
        max_depth_label.setToolTip("Максимальное количество итераций для вокселизации (больше = выше детализация, медленнее)")
        parent.max_depth_slider = QSlider(Qt.Horizontal)
        parent.max_depth_slider.setMinimum(5)
        parent.max_depth_slider.setMaximum(20)
        parent.max_depth_slider.setValue(10)
        parent.max_depth_slider.setToolTip("5–20: большие значения увеличивают детализацию, но замедляют обработку")
        parent.max_depth_value = QLabel(f"{parent.max_depth_slider.value()}")
        parent.max_depth_slider.valueChanged.connect(lambda: parent.max_depth_value.setText(f"{parent.max_depth_slider.value()}"))
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(max_depth_label)
        depth_layout.addWidget(parent.max_depth_slider)
        depth_layout.addWidget(parent.max_depth_value)
        settings_layout.addLayout(depth_layout)

        # Curvature-Based
        parent.curvature_based = QCheckBox("Curvature-Based")
        parent.curvature_based.setObjectName("curvatureBased")
        parent.curvature_based.setToolTip("Включить адаптивную вокселизацию на основе кривизны")
        settings_layout.addWidget(parent.curvature_based)

        # Minimal Support
        parent.minimal_support = QCheckBox("Minimal Support")
        parent.minimal_support.setObjectName("minimalSupport")
        parent.minimal_support.setToolTip("Заполнять только для минимальной структурной поддержки")
        settings_layout.addWidget(parent.minimal_support)

        # Scale Factor
        scale_factor_label = QLabel("Scale Factor")
        scale_factor_label.setToolTip("Коэффициент масштабирования модели (0.1–10.0)")
        parent.scale_factor = QSlider(Qt.Horizontal)
        parent.scale_factor.setRange(10, 1000)
        parent.scale_factor.setValue(100)
        parent.scale_value = QLabel("1.0")
        parent.scale_factor.valueChanged.connect(lambda v: parent.scale_value.setText(f"{v/100:.1f}"))
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(scale_factor_label)
        scale_layout.addWidget(parent.scale_factor)
        scale_layout.addWidget(parent.scale_value)
        settings_layout.addLayout(scale_layout)

        # Instruction Style
        instruction_label = QLabel("Instruction Style")
        instruction_label.setToolTip("Управляет способом группировки шагов сборки")
        parent.instruction_style = QComboBox()
        parent.instruction_style.addItems(["Fast Grouping", "Detailed Grouping"])
        parent.instruction_style.setCurrentIndex(0)
        parent.instruction_style.setToolTip("Быстрая группировка: меньше шагов (метод соединений), Детальная группировка: точные шаги (метод DBSCAN)")
        settings_layout.addWidget(instruction_label)
        settings_layout.addWidget(parent.instruction_style)

        # Placement Method
        placement_label = QLabel("Placement Method")
        placement_label.setToolTip("Выбирает алгоритм размещения LEGO-кирпичей")
        parent.placement_method = QComboBox()
        parent.placement_method.addItems(["Greedy (Fast)", "Simulated Annealing", "Branch and Bound"])
        parent.placement_method.setCurrentIndex(0)
        parent.placement_method.setToolTip("Жадный: быстрый и простой, Симулированный отжиг: сбалансированная оптимизация, Ветвление и границы: точный, но медленный")
        settings_layout.addWidget(placement_label)
        settings_layout.addWidget(parent.placement_method)

        # Toggles (Use Colors, Render Steps, Generate Instructions)
        toggles_widget = QWidget()
        toggles_layout = QHBoxLayout(toggles_widget)
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        toggles_layout.setSpacing(10)

        # Use Colors
        enable_colors_label = QLabel("Colors")
        parent.use_colors = QPushButton()
        parent.use_colors.setObjectName("useColors")
        parent.use_colors.setCheckable(True)
        parent.use_colors.setChecked(True)
        parent.use_colors.setToolTip("Включено: использовать цвета модели для кирпичей, Выключено: использовать серые кирпичи")
        toggles_layout.addWidget(enable_colors_label)
        toggles_layout.addWidget(parent.use_colors)

        # Render Steps
        render_steps_label = QLabel("Steps")
        parent.render_steps = QPushButton()
        parent.render_steps.setObjectName("renderSteps")
        parent.render_steps.setCheckable(True)
        parent.render_steps.setChecked(True)
        parent.render_steps.setToolTip("Включено: рендерить шаги сборки, Выключено: пропустить рендеринг шагов")
        toggles_layout.addWidget(render_steps_label)
        toggles_layout.addWidget(parent.render_steps)

        # Generate Instructions
        generate_instructions_label = QLabel("Instructions")
        parent.generate_instructions = QPushButton()
        parent.generate_instructions.setObjectName("generateInstructions")
        parent.generate_instructions.setCheckable(True)
        parent.generate_instructions.setChecked(True)
        parent.generate_instructions.setToolTip("Включено: создавать инструкции, Выключено: пропустить создание инструкций")
        toggles_layout.addWidget(generate_instructions_label)
        toggles_layout.addWidget(parent.generate_instructions)

        toggles_layout.addStretch(1)
        settings_layout.addWidget(toggles_widget)

        # Parallel Processing
        parent.parallel_processing = QCheckBox("Parallel Processing")
        parent.parallel_processing.setObjectName("parallelProcessing")
        parent.parallel_processing.setToolTip("Использовать несколько ядер для ускорения генерации инструкций")
        settings_layout.addWidget(parent.parallel_processing)

        # Stats
        stats_layout = QFormLayout()
        stats_layout.setSpacing(10)
        stats_layout.setLabelAlignment(Qt.AlignLeft)
        bricks_label = QLabel("Bricks:")
        parent.bricks_value = QLabel("0")
        stats_layout.addRow(bricks_label, parent.bricks_value)
        stats_layout.setAlignment(parent.bricks_value, Qt.AlignRight)
        hline1 = QFrame()
        hline1.setFrameShape(QFrame.HLine)
        hline1.setFrameShadow(QFrame.Sunken)
        stats_layout.addRow(hline1)
        time_label = QLabel("Time:")
        parent.time_value = QLabel("0s")
        stats_layout.addRow(time_label, parent.time_value)
        stats_layout.setAlignment(parent.time_value, Qt.AlignRight)
        hline2 = QFrame()
        hline2.setFrameShape(QFrame.HLine)
        hline2.setFrameShadow(QFrame.Sunken)
        stats_layout.addRow(hline2)
        complexity_label = QLabel("Complexity:")
        parent.complexity_value = QLabel("N/A")
        stats_layout.addRow(complexity_label, parent.complexity_value)
        stats_layout.setAlignment(parent.complexity_value, Qt.AlignRight)
        hline3 = QFrame()
        hline3.setFrameShape(QFrame.HLine)
        hline3.setFrameShadow(QFrame.Sunken)
        stats_layout.addRow(hline3)
        settings_layout.addLayout(stats_layout)

        # Output Path
        output_path_label = QLabel("Output Path")
        output_path_top_layout = QHBoxLayout()
        output_path_top_layout.addWidget(output_path_label)
        output_path_top_layout.setSpacing(20)
        parent.output_path = QLineEdit(DEFAULT_OUTPUT_PATH)
        parent.output_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        parent.output_path.setToolTip("Путь для сохранения результатов")
        output_path_top_layout.addWidget(parent.output_path, stretch=1)
        output_path_button = QPushButton("Browse")
        output_path_button.setObjectName("outputPathButton")
        output_path_button.setIcon(QIcon("icons/folder2.svg"))
        output_path_button.setIconSize(QSize(*ICON_SIZE))
        output_path_button.setFixedSize(*OUTPUT_PATH_BUTTON_SIZE)
        output_path_button.clicked.connect(parent.browse_output)
        output_path_button.setToolTip("Выбрать папку для сохранения файлов")
        output_path_layout = QVBoxLayout()
        output_path_layout.addLayout(output_path_top_layout)
        output_path_layout.addSpacing(20)
        output_path_layout.addWidget(output_path_button)
        output_path_layout.setAlignment(output_path_button, Qt.AlignRight)
        settings_layout.addSpacing(20)
        settings_layout.addLayout(output_path_layout)

        # Export Options
        parent.export_voxelized = QCheckBox("Export Voxelized STL")
        parent.export_voxelized.setObjectName("exportVoxelized")
        parent.export_voxelized.setChecked(True)
        parent.export_voxelized.setToolTip("Экспортировать вокселизованную модель в формате STL")
        settings_layout.addWidget(parent.export_voxelized)

        parent.export_unique_bricks = QCheckBox("Export Unique Bricks STL")
        parent.export_unique_bricks.setObjectName("exportUniqueBricks")
        parent.export_unique_bricks.setChecked(True)
        parent.export_unique_bricks.setToolTip("Экспортировать уникальные кирпичи в формате STL")
        settings_layout.addWidget(parent.export_unique_bricks)

        # Step Image Size
        step_image_size_label = QLabel("Step Image Size (px)")
        step_image_size_label.setToolTip("Размер изображений шагов в пикселях (200–600)")
        parent.step_image_size = QSlider(Qt.Horizontal)
        parent.step_image_size.setRange(200, 600)
        parent.step_image_size.setValue(300)
        parent.step_image_value = QLabel("300")
        parent.step_image_size.valueChanged.connect(lambda v: parent.step_image_value.setText(str(v)))
        step_image_layout = QHBoxLayout()
        step_image_layout.addWidget(step_image_size_label)
        step_image_layout.addWidget(parent.step_image_size)
        step_image_layout.addWidget(parent.step_image_value)
        settings_layout.addLayout(step_image_layout)

        # Log Level
        log_level_label = QLabel("Log Level")
        log_level_label.setToolTip("Уровень логирования для отладки")
        parent.log_level = QComboBox()
        parent.log_level.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])
        parent.log_level.setCurrentText("INFO")
        # Исправляем обработчик
        def set_log_level(level):
            log_levels = {
                "INFO": logging.INFO,
                "DEBUG": logging.DEBUG,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR
            }
            logging.getLogger().setLevel(log_levels[level])
        parent.log_level.currentTextChanged.connect(set_log_level)
        parent.log_level.setToolTip("INFO: стандартный, DEBUG: подробный, WARNING: предупреждения, ERROR: только ошибки")
        settings_layout.addWidget(log_level_label)
        settings_layout.addWidget(parent.log_level)

                # Allow Top Layer (оставляем в Cubs, но уточним у тебя)
        parent.allow_top_layer = QCheckBox("Allow Top Layer")
        parent.allow_top_layer.setObjectName("allowTopLayer")
        parent.allow_top_layer.setToolTip("Разрешить размещение кирпичей сверху с боковой поддержкой")
        settings_layout.addWidget(parent.allow_top_layer)

        settings_layout.addStretch(1)
        settings_scroll.setWidget(settings_widget)
        self.addTab(settings_scroll, "Settings")

        # Вкладка Cubs
        cubs_scroll = QScrollArea()
        cubs_scroll.setWidgetResizable(True)
        cubs_widget = QWidget()
        cubs_widget.setMinimumHeight(SETTINGS_PANEL_MIN_HEIGHT)
        cubs_layout = QVBoxLayout(cubs_widget)
        cubs_layout.setContentsMargins(20, 20, 20, 20)
        cubs_layout.setSpacing(20)

        # Brick Sizes
        brick_sizes_label = QLabel("Allowed Brick Sizes")
        brick_sizes_label.setToolTip("Выберите допустимые размеры кирпичей")
        parent.brick_sizes = QListWidget()
        parent.brick_sizes.setObjectName("brick_sizes")
        parent.brick_sizes.setSelectionMode(QListWidget.MultiSelection)
        for w, h, d, brick_type in BRICK_SIZES:
            item = QListWidgetItem(f"{w}x{h}x{d} ({brick_type})")
            parent.brick_sizes.addItem(item)
            item.setSelected(True)
        parent.brick_sizes.setToolTip("Выберите допустимые размеры кирпичей; комбинируется с выбором стиля кирпичей")
        cubs_layout.addWidget(brick_sizes_label)
        cubs_layout.addWidget(parent.brick_sizes)

        cubs_scroll.setWidget(cubs_widget)
        self.addTab(cubs_scroll, "Cubs")

class ActionButtons(QWidget):
    def __init__(self, parent):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setSpacing(10)

        parent.generate_button = QPushButton("Generation")
        parent.generate_button.setObjectName("generateButton")
        parent.generate_button.setIcon(QIcon("icons/generate_cube.png"))
        parent.generate_button.setIconSize(QSize(*ICON_SIZE))
        parent.generate_button.setFixedSize(*ACTION_BUTTON_SIZE)
        parent.generate_button.setEnabled(False)
        parent.generate_button.clicked.connect(parent.toggle_generation)
        parent.generate_button.setToolTip("Сгенерировать модель LEGO из загруженного файла")

        load_model_button = QPushButton()
        load_model_button.setObjectName("loadModelButton")
        load_model_button.setIcon(QIcon("icons/folder.svg"))
        load_model_button.setIconSize(QSize(*ICON_SIZE))
        load_model_button.apply_shadow = self.apply_shadow
        load_model_button.apply_shadow()
        load_model_button.setFixedSize(*SMALL_BUTTON_SIZE)
        load_model_button.clicked.connect(parent.load_file)
        load_model_button.setToolTip("Загрузить 3D-модель (STL или OBJ)")

        instruction_button = QPushButton()
        instruction_button.setObjectName("instructionButton")
        instruction_button.setIcon(QIcon("icons/document.png"))
        instruction_button.setIconSize(QSize(*ICON_SIZE))
        instruction_button.apply_shadow = self.apply_shadow
        instruction_button.apply_shadow()
        instruction_button.setFixedSize(*SMALL_BUTTON_SIZE)
        instruction_button.clicked.connect(parent.open_pdf)
        instruction_button.setToolTip("Открыть сгенерированные инструкции в формате PDF")

        steps_button = QPushButton()
        steps_button.setObjectName("stepsButton")
        steps_button.setIcon(QIcon("icons/steps.svg"))
        steps_button.setIconSize(QSize(*ICON_SIZE))
        steps_button.apply_shadow = self.apply_shadow
        steps_button.apply_shadow()
        steps_button.setFixedSize(*SMALL_BUTTON_SIZE)
        steps_button.clicked.connect(parent.toggle_thumbnail)
        steps_button.setToolTip("Показать/скрыть миниатюры пошаговой сборки")

        layout.addWidget(parent.generate_button)
        layout.addWidget(load_model_button)
        layout.addWidget(instruction_button)
        layout.addWidget(steps_button)

    def apply_shadow(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(SHADOW_BLUR_RADIUS)
        shadow.setXOffset(SHADOW_X_OFFSET)
        shadow.setYOffset(SHADOW_Y_OFFSET)
        shadow.setColor(QColor(12, 12, 13, SHADOW_ALPHA))
        self.setGraphicsEffect(shadow)

class ProgressLogs(QTabWidget):
    def __init__(self, parent):
        super().__init__()
        self.setObjectName("progressLogs")
        self.setMinimumSize(MODEL_WINDOW_MIN_WIDTH, 106)
        self.apply_shadow()
        self.setup_tabs(parent)

    def apply_shadow(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(SHADOW_BLUR_RADIUS)
        shadow.setXOffset(SHADOW_X_OFFSET)
        shadow.setYOffset(SHADOW_Y_OFFSET)
        shadow.setColor(QColor(12, 12, 13, SHADOW_ALPHA))
        self.setGraphicsEffect(shadow)

    def setup_tabs(self, parent):
        progress_tab = QWidget()
        logs_tab = QWidget()
        self.addTab(progress_tab, "Progress")
        self.addTab(logs_tab, "Logs")

        progress_layout = QVBoxLayout(progress_tab)
        progress_layout.setAlignment(Qt.AlignVCenter)
        parent.progress = QProgressBar()
        parent.progress.setTextVisible(True)
        parent.progress.setFixedHeight(PROGRESS_HEIGHT)
        parent.progress_label = QLabel("0%")
        parent.progress_label.setAlignment(Qt.AlignCenter)
        parent.progress_stage = QLabel("Waiting to start...")
        progress_layout.addWidget(parent.progress)
        progress_layout.addWidget(parent.progress_label)
        progress_layout.addWidget(parent.progress_stage)

        logs_layout = QVBoxLayout(logs_tab)
        parent.log_display = QTextEdit()
        parent.log_display.setObjectName("logDisplay")
        parent.log_display.setReadOnly(True)
        logs_layout.addWidget(parent.log_display)
        parent.log_display.apply_shadow = self.apply_shadow
        parent.log_display.apply_shadow()