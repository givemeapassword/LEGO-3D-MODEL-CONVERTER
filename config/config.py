from typing import Dict, Tuple, List
from reportlab.lib.pagesizes import A4

# LEGO-кирпичи
STUD_SIZE: float = 7.8 # Минимальная единица высоты (пластина)
VOXEL_SIZE_DEFAULT: float = 3.2
MIN_OVERLAP = 8.0
BRICK_HEIGHTS = {"brick": 9.6, "plate": 3.2, "tile": 3.2}  # Высота одного слоя кирпича в мм
LEGO_COLORS: List[str] = [
    "#FF0000", "#0000FF", "#FFFF00", "#008000", "#000000", "#FFFFFF",
    "#FFA500", "#800080", "#A52A2A", "#008080", "#FF69B4", "#645452",
    "#C0C0C0", "#FFD700", "#2F4F4F", "#FFC0CB", "#87CEEB", "#98FB98", "#F5F5DC"
]

BRICK_PROPERTIES = {
    "brick": {"stability": 1.0, "surface_score": 0.5, "support_score": 1.0},  # Обычный кирпич (например, 2x4x2)
    "plate": {"stability": 0.7, "surface_score": 1.0, "support_score": 0.3},  # Плоская пластина (например, 1x4x1)
    "tile": {"stability": 0.5, "surface_score": 1.2, "support_score": 0.1}    # Тонкая плитка (например, 1x2x0.33)
}

BRICK_SIZES = [
    # Стандартные кирпичи (brick) с высотой 1
    (1, 1, 1, "brick"),  # Добавлен маленький кирпич 1x1 для мелкой детализации
    (1, 2, 1, "brick"),
    (1, 3, 1, "brick"),
    (1, 4, 1, "brick"),
    (1, 6, 1, "brick"),  # Добавлен длинный кирпич 1x6
    (2, 2, 1, "brick"),
    (2, 3, 1, "brick"),  # Добавлен 2x3
    (2, 4, 1, "brick"),
    (2, 6, 1, "brick"),  # Добавлен длинный кирпич 2x6
    (3, 3, 1, "brick"),  # Добавлен 3x3 для квадратных участков
    (4, 4, 1, "brick"),
    
    # Высокие кирпичи (brick) с высотой 2
    (1, 1, 2, "brick"),  # Добавлен маленький высокий кирпич
    (1, 2, 2, "brick"),
    (2, 2, 2, "brick"),
    (2, 4, 2, "brick"),  # Добавлен высокий и длинный кирпич
    
    # Пластины (plate) с высотой 1/3 кирпича
    (1, 1, 1, "plate"),  # Маленькая пластина для мелкой детализации
    (1, 2, 1, "plate"),
    (1, 3, 1, "plate"),
    (1, 4, 1, "plate"),
    (1, 6, 1, "plate"),  # Длинная пластина
    (2, 2, 1, "plate"),
    (2, 3, 1, "plate"),
    (2, 4, 1, "plate"),
    (2, 6, 1, "plate"),  # Длинная пластина
    (3, 3, 1, "plate"),  # Квадратная пластина
    (4, 2, 1, "plate"),
    (4, 4, 1, "plate"),
    
    # Плитки (tile) без шипов, высота 1/3 кирпича
    (1, 1, 1, "tile"),   # Добавлена маленькая плитка
    (1, 2, 1, "tile"),
    (1, 4, 1, "tile"),   # Добавлена длинная плитка
    (2, 2, 1, "tile"),
    (2, 4, 1, "tile"),   # Длинная плитка
    
]

def get_brick_height(brick_type):
    return BRICK_HEIGHTS.get(brick_type, 9.6)

# Настройки GUI
WINDOW_TITLE: str = "Lego Builder Pro"
WINDOW_GEOMETRY: Tuple[int, int, int, int] = (100, 100, 1600, 900)
DEFAULT_OUTPUT_PATH: str = "output"

# Параметры вокселизации и обработки
VOXEL_SIZE_RANGE: Tuple[float, float] = (0.1, 5.0)
VOXEL_SIZE_DEFAULT: float = 1.0
MAX_DEPTH_RANGE: Tuple[int, int] = (1, 20)
MAX_DEPTH_DEFAULT: int = 6
SCALE_FACTOR_RANGE: Tuple[float, float] = (0.1, 10.0)
SCALE_FACTOR_DEFAULT: float = 1.0
SCALE_FACTOR_STEP: float = 0.1
CLUSTERING_METHODS: List[str] = ["greedy", "dbscan", "simulated_annealing", "branch_and_bound"]
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".stl", ".obj")
DEFAULT_RADIUS: float = 5.0  # Радиус для измерения кривизны
MIN_RADIUS = 1.0      # Минимальный радиус для мелких деталей
MAX_RADIUS = 20.0     # Максимальный радиус для больших моделей
RADIUS_FRACTION = 0.1
EPSILON: float = 1e-6  # Малое значение для предотвращения деления на ноль
MAX_ITER_MULTIPLIER: int = 2  # Множитель для увеличения max_iter

# Параметры камеры и взаимодействия
CAMERA_DISTANCE_FACTOR: float = 5.0  # Множитель расстояния камеры от модели
DEFAULT_VIEW_UP: Tuple[float, float, float] = (0, 0, 1)  # Направление "вверх" по умолчанию
VIEW_CONFIGS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "top": ((0, 0, 1), (0, 1, 0)),
    "bottom": ((0, 0, -1), (0, 1, 0)),
    "front": ((0, 1, 0), (0, 0, 1)),
    "back": ((0, -1, 0), (0, 0, 1)),
    "right": ((1, 0, 0), (0, 0, 1)),
    "left": ((-1, 0, 0), (0, 0, 1)),
    "iso-top-front-right": ((1, 1, 1), DEFAULT_VIEW_UP),
    "iso-top-front-left": ((-1, 1, 1), DEFAULT_VIEW_UP),
    "iso-top-back-right": ((1, -1, 1), DEFAULT_VIEW_UP),
    "iso-top-back-left": ((-1, -1, 1), DEFAULT_VIEW_UP),
    "iso-bottom-front-right": ((1, 1, -1), DEFAULT_VIEW_UP),
    "iso-bottom-front-left": ((-1, 1, -1), DEFAULT_VIEW_UP),
    "iso-bottom-back-right": ((1, -1, -1), DEFAULT_VIEW_UP),
    "iso-bottom-back-left": ((-1, -1, -1), DEFAULT_VIEW_UP),
    "edge-top-front": ((0, 1, 1), DEFAULT_VIEW_UP),
    "edge-top-back": ((0, -1, 1), DEFAULT_VIEW_UP),
    "edge-top-right": ((1, 0, 1), DEFAULT_VIEW_UP),
    "edge-top-left": ((-1, 0, 1), DEFAULT_VIEW_UP),
    "edge-bottom-front": ((0, 1, -1), DEFAULT_VIEW_UP),
    "edge-bottom-back": ((0, -1, -1), DEFAULT_VIEW_UP),
    "edge-bottom-right": ((1, 0, -1), DEFAULT_VIEW_UP),
    "edge-bottom-left": ((-1, 0, -1), DEFAULT_VIEW_UP),
    "edge-front-right": ((1, 1, 0), DEFAULT_VIEW_UP),
    "edge-front-left": ((-1, 1, 0), DEFAULT_VIEW_UP),
    "edge-back-right": ((1, -1, 0), DEFAULT_VIEW_UP),
    "edge-back-left": ((-1, -1, 0), DEFAULT_VIEW_UP),
}

# Параметры рендеринга
LIGHT_POSITION: Tuple[float, float, float] = (10, 10, 10)
LIGHT_INTENSITY_TOP: float = 0.5  # Интенсивность света сверху
LIGHT_INTENSITY_SIDES: float = 0.5  # Интенсивность боковых источников света
LIGHT_INTENSITY_AMBIENT: float = 0.3  # Интенсивность фонового света
LIGHT_DISTANCE_FACTOR: float = 2.0  # Множитель расстояния для источников света
FLOOR_OPACITY: float = 1.0  # Прозрачность пола
FLOOR_COLOR: str = "#333333"  # Цвет пола
FLOOR_EDGE_COLOR: str = "#FFFFFF"  # Цвет линий сетки пола
CUBE_EDGE_COLOR: str = "#666666"  # Цвет линий граней кубов
CUBE_OPACITY: float = 1.0  # Прозрачность кубов
ORIGINAL_MESH_COLOR: str = "#FFFF40"  # Цвет оригинальной модели
ORIGINAL_MESH_OPACITY: float = 1.0  # Прозрачность оригинальной модели
RENDER_CAMERA_POSITION: str = "iso"
RENDER_LIGHT_POSITION: Tuple[float, float, float] = (10, 10, 10)

# Настройки PDF
PDF_PAGE_SIZE: Tuple[float, float] = A4
STEP_CIRCLE_RADIUS: float = 20
STEP_CIRCLE_POS: Tuple[float, float] = (50, PDF_PAGE_SIZE[1] - 50)
STEP_IMAGE_POS: Tuple[float, float, float, float] = (100, PDF_PAGE_SIZE[1] / 2 - 150, 300, 300)  # x, y, width, height
DESCRIPTION_POS: Tuple[float, float] = (100, PDF_PAGE_SIZE[1] / 2 - 180)
TEMP_IMAGE_DIR: str = "instructions_images"


## GUI.PY
# Константы для окна
WINDOW_WIDTH = 1700
WINDOW_HEIGHT = 900
BASE_WIDTH = 1700

# Константы размеров виджетов
HEADER_HEIGHT = 60
MODEL_WINDOW_MIN_WIDTH = 1105
SETTINGS_PANEL_MIN_WIDTH = 320
SETTINGS_PANEL_MIN_HEIGHT = 700
ACTION_BUTTON_SIZE = (240, 80)  # (width, height)
SMALL_BUTTON_SIZE = (80, 80)    # Для кнопок load_model, instruction, steps
THUMBNAIL_WIDTH = 220
LOGO_SIZE = (42, 40)           # (width, height)
ICON_SIZE = (36, 36)           # Для большинства иконок
THUMBNAIL_ICON_SIZE = (64, 64) # Для иконок в списке thumbnails
BUTTON_GROUP_SIZE = (60, 60)   # Для кнопок минимизации и закрытия
OUTPUT_PATH_BUTTON_SIZE = (150, 40)
TOGGLE_BUTTON_SIZE = (60, 34)  # Для переключателя use_colors
PROGRESS_HEIGHT = 13
TAB_WIDTH = 245
TAB_HEIGHT = 48
PROGRESS_TAB_WIDTH = 150
PROGRESS_TAB_HEIGHT = 25

# Константы для слайдеров
VOXEL_SIZE_MIN = 10
VOXEL_SIZE_MAX = 500
SCALE_FACTOR_MIN = 10
SCALE_FACTOR_MAX = 500

# Константы анимации и таймеров
SNACKBAR_ANIMATION_DURATION = 400  # мс
SNACKBAR_DISPLAY_DURATION = 3000  # мс
CAMERA_ANIMATION_STEPS = 30
CAMERA_ANIMATION_INTERVAL = 16    # мс
LOG_UPDATE_INTERVAL = 500         # мс
PROGRESS_UPDATE_INTERVAL = 0.5    # сек

# Константы теней
SHADOW_BLUR_RADIUS = 4
SHADOW_X_OFFSET = 0
SHADOW_Y_OFFSET = 1
SHADOW_ALPHA = int(0.05 * 255)

# Дополнительные константы
ROUNDING_RADIUS = 20              # Радиус скругления углов