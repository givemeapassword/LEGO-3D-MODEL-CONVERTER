import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt
import numpy as np
import logging

class ViewCube(QtInteractor):
    def __init__(self, parent=None, set_view_callback=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # Прозрачный фон для виджета
        self.setStyleSheet("background: transparent;")  # Устанавливаем прозрачный фон
        self.setFixedSize(150, 150)  # Увеличиваем размер окна до 150x150
        self.set_view_callback = set_view_callback
        self.hovered_region = None
        self.is_dragging = False
        self.last_pos = None
        self.face_centers = {
        "top": [0, 0, 1],
        "bottom": [0, 0, -1],
        "front": [0, 1, 0],
        "back": [0, -1, 0],
        "right": [1, 0, 0],
        "left": [-1, 0, 0],
        }   

        self.set_background("#C0C0C0")

        # Создаём 3D-куб
        self.cube = pv.Cube(center=(0, 0, 0), x_length=2, y_length=2, z_length=2)
        self.cube_actor = self.add_mesh(self.cube, color="#4c4c4c", show_edges=True, edge_color="#333333", opacity=1.0, lighting=True, ambient=0.5, diffuse=0.5, specular=0.0)

        # Определяем грани куба и их нормали
        self.faces = {
            "top": {"normal": [0, 0, 1], "view": "top"},
            "bottom": {"normal": [0, 0, -1], "view": "bottom"},
            "front": {"normal": [0, 1, 0], "view": "front"},
            "back": {"normal": [0, -1, 0], "view": "back"},
            "right": {"normal": [1, 0, 0], "view": "right"},
            "left": {"normal": [-1, 0, 0], "view": "left"},
        }

        # Определяем углы куба (каждый угол связан с тремя гранями)
        self.corners = {
            "top-front-right": {"direction": [1, 1, 1], "views": ["top", "front", "right"]},
            "top-front-left": {"direction": [-1, 1, 1], "views": ["top", "front", "left"]},
            "top-back-right": {"direction": [1, -1, 1], "views": ["top", "back", "right"]},
            "top-back-left": {"direction": [-1, -1, 1], "views": ["top", "back", "left"]},
            "bottom-front-right": {"direction": [1, 1, -1], "views": ["bottom", "front", "right"]},
            "bottom-front-left": {"direction": [-1, 1, -1], "views": ["bottom", "front", "left"]},
            "bottom-back-right": {"direction": [1, -1, -1], "views": ["bottom", "back", "right"]},
            "bottom-back-left": {"direction": [-1, -1, -1], "views": ["bottom", "back", "left"]},
        }

        # Определяем рёбра куба (каждое ребро связано с двумя гранями)
        self.edges = {
            "top-front": {"direction": [0, 1, 1], "views": ["top", "front"]},
            "top-back": {"direction": [0, -1, 1], "views": ["top", "back"]},
            "top-right": {"direction": [1, 0, 1], "views": ["top", "right"]},
            "top-left": {"direction": [-1, 0, 1], "views": ["top", "left"]},
            "bottom-front": {"direction": [0, 1, -1], "views": ["bottom", "front"]},
            "bottom-back": {"direction": [0, -1, -1], "views": ["bottom", "back"]},
            "bottom-right": {"direction": [1, 0, -1], "views": ["bottom", "right"]},
            "bottom-left": {"direction": [-1, 0, -1], "views": ["bottom", "left"]},
            "front-right": {"direction": [1, 1, 0], "views": ["front", "right"]},
            "front-left": {"direction": [-1, 1, 0], "views": ["front", "left"]},
            "back-right": {"direction": [1, -1, 0], "views": ["back", "right"]},
            "back-left": {"direction": [-1, -1, 0], "views": ["back", "left"]},
        }

        # Настраиваем камеру
        self.reset_camera()
        self.camera_position = "iso"  # Изометрический вид
        self.camera.SetPosition(6, 6, 6)  # Позиция камеры
        self.camera.SetFocalPoint(0, 0, 0)  # Точка фокуса в центре куба
        self.camera.SetViewUp(0, 0, 1)  # Направление "вверх"
        self.camera.Zoom(1.2)  # Уменьшаем масштаб
        logging.debug(f"Camera after setup - Position: {self.camera.GetPosition()}, Focal Point: {self.camera.GetFocalPoint()}, View Up: {self.camera.GetViewUp()}")

        self.remove_all_lights()
        self.add_light(pv.Light(
            color="white",
            intensity=0.8,
            positional=False,
            show_actor=False
        ))
        self.add_light(pv.Light(
            position=(5, 5, 5),
            color="white",
            intensity=0.3,
            positional=True,
            show_actor=False
        ))

        self.labels = {}
        # Флаг для отслеживания, были ли метки уже созданы
        self.labels_created = False
        self.cube_actor.GetProperty().SetColor(0.3, 0.3, 0.3)  # Исходный цвет (темно-серый #4c4c4c)
        self.cube_actor.GetProperty().SetOpacity(1.0)  # Исходная непрозрачность


    def mouseMoveEvent(self, event):
            if self.last_pos and self.is_dragging:
                # Вращение куба при перетаскивании
                delta_x = event.pos().x() - self.last_pos.x()
                delta_y = event.pos().y() - self.last_pos.y()
                self.camera.Azimuth(delta_x * 0.5)
                self.camera.Elevation(-delta_y * 0.5)
                self.last_pos = event.pos()
                self.update()
                self.update_labels()
                if self.set_view_callback:
                    camera_pos = np.array(self.camera.GetPosition())
                    camera_focal_point = np.array(self.camera.GetFocalPoint())
                    camera_view_up = np.array(self.camera.GetViewUp())
                    self.set_view_callback("custom", camera_pos, camera_focal_point, camera_view_up)
                # Сбрасываем подсветку во время перетаскивания
                self.highlight_region(None, None)
            else:
                # Определяем регион под курсором
                camera_pos = np.array(self.camera.GetPosition())
                camera_focal_point = np.array(self.camera.GetFocalPoint())
                direction = camera_focal_point - camera_pos
                direction = direction / np.linalg.norm(direction)

                # Проверяем грани, углы, рёбра
                closest_face, max_face_dot = None, -1
                closest_corner, max_corner_dot = None, -1
                closest_edge, max_edge_dot = None, -1

                for face_name, face_data in self.faces.items():
                    normal = np.array(face_data["normal"])
                    dot = np.dot(-direction, normal)
                    if dot > max_face_dot:
                        max_face_dot = dot
                        closest_face = face_name

                for corner_name, corner_data in self.corners.items():
                    corner_dir = np.array(corner_data["direction"]) / np.linalg.norm(corner_data["direction"])
                    dot = np.dot(-direction, corner_dir)
                    if dot > max_corner_dot:
                        max_corner_dot = dot
                        closest_corner = corner_name

                for edge_name, edge_data in self.edges.items():
                    edge_dir = np.array(edge_data["direction"]) / np.linalg.norm(edge_data["direction"])
                    dot = np.dot(-direction, edge_dir)
                    if dot > max_edge_dot:
                        max_edge_dot = dot
                        closest_edge = edge_name

                # Пороги для подсветки
                threshold = 0.9
                if max_face_dot > threshold and max_face_dot > max_corner_dot and max_face_dot > max_edge_dot:
                    self.highlight_region("face", closest_face)
                elif max_corner_dot > threshold and max_corner_dot > max_face_dot and max_corner_dot > max_edge_dot:
                    self.highlight_region("corner", closest_corner)
                elif max_edge_dot > threshold and max_edge_dot > max_face_dot and max_edge_dot > max_corner_dot:
                    self.highlight_region("edge", closest_edge)
                else:
                    self.highlight_region(None, None)

            super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.is_dragging = False
                self.last_pos = event.pos()
                # Не сбрасываем hovered_region здесь, чтобы сохранить подсветку до движения
            super().mousePressEvent(event)

    def highlight_region(self, region_type, region_name):
            """Подсвечивает указанную область или сбрасывает подсветку."""
            if self.hovered_region != (region_type, region_name):
                # Сбрасываем цвет и прозрачность
                self.cube_actor.GetProperty().SetColor(0.3, 0.3, 0.3)  # #4c4c4c в RGB
                self.cube_actor.GetProperty().SetOpacity(1.0)

                # Применяем новую подсветку
                if region_type == "face":
                    self.cube_actor.GetProperty().SetColor(0.0, 0.6, 1.0)  # Голубой
                    self.cube_actor.GetProperty().SetOpacity(0.7)
                    logging.debug(f"Highlighting face: {region_name}")
                elif region_type == "corner":
                    self.cube_actor.GetProperty().SetColor(0.0, 0.7, 1.0)
                    self.cube_actor.GetProperty().SetOpacity(0.7)
                    logging.debug(f"Highlighting corner: {region_name}")
                elif region_type == "edge":
                    self.cube_actor.GetProperty().SetColor(0.0, 0.65, 1.0)
                    self.cube_actor.GetProperty().SetOpacity(0.7)
                    logging.debug(f"Highlighting edge: {region_name}")

                self.hovered_region = (region_type, region_name)
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.last_pos = None
            if not self.is_dragging:
                # Логика клика остается прежней
                camera_pos = np.array(self.camera.GetPosition())
                camera_focal_point = np.array(self.camera.GetFocalPoint())
                direction = camera_focal_point - camera_pos
                direction = direction / np.linalg.norm(direction)

                closest_face, max_face_dot = None, -1
                closest_corner, max_corner_dot = None, -1
                closest_edge, max_edge_dot = None, -1

                for face_name, face_data in self.faces.items():
                    normal = np.array(face_data["normal"])
                    dot = np.dot(-direction, normal)
                    if dot > max_face_dot:
                        max_face_dot = dot
                        closest_face = face_name

                for corner_name, corner_data in self.corners.items():
                    corner_dir = np.array(corner_data["direction"]) / np.linalg.norm(corner_data["direction"])
                    dot = np.dot(-direction, corner_dir)
                    if dot > max_corner_dot:
                        max_corner_dot = dot
                        closest_corner = corner_name

                for edge_name, edge_data in self.edges.items():
                    edge_dir = np.array(edge_data["direction"]) / np.linalg.norm(edge_data["direction"])
                    dot = np.dot(-direction, edge_dir)
                    if dot > max_edge_dot:
                        max_edge_dot = dot
                        closest_edge = edge_name

                threshold = 0.9
                if max_face_dot > threshold and max_face_dot > max_corner_dot and max_face_dot > max_edge_dot:
                    if self.set_view_callback:
                        self.set_view_callback(self.faces[closest_face]["view"])
                elif max_corner_dot > threshold and max_corner_dot > max_face_dot and max_corner_dot > max_edge_dot:
                    if self.set_view_callback:
                        views = self.corners[closest_corner]["views"]
                        self.set_view_callback(f"iso-{views[0]}-{views[1]}-{views[2]}")
                elif max_edge_dot > threshold and max_edge_dot > max_face_dot and max_edge_dot > max_corner_dot:
                    if self.set_view_callback:
                        views = self.edges[closest_edge]["views"]
                        self.set_view_callback(f"edge-{views[0]}-{views[1]}")
            # Сбрасываем подсветку после клика, если не перетаскивали
            if not self.is_dragging:
                self.highlight_region(None, None)
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        # Сбрасываем подсветку при уходе мыши
        self.highlight_region(None, None)
        super().leaveEvent(event)
        
    def update_labels(self):
        """Обновляет позиции подписей на гранях куба."""
        if not self.labels_created:
            return  # Если метки ещё не созданы, ничего не делаем

        for face_name, center in self.face_centers.items():
            renderer = self.renderer
            renderer.SetWorldPoint(center[0], center[1], center[2], 1.0)
            renderer.WorldToDisplay()
            display_point = renderer.GetDisplayPoint()
            if face_name in self.labels:
                self.labels[face_name].SetPosition(display_point[0], display_point[1])

    def leaveEvent(self, event):
        # Сбрасываем подсветку при уходе мыши
        self.highlight_region(None, None)
        super().leaveEvent(event)