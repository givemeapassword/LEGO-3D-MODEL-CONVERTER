from PyQt5.QtCore import QObject, pyqtSignal

class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(list, list, str)  # Добавлен путь к PDF
    error = pyqtSignal(str)