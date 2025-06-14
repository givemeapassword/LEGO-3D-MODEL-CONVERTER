# src/gui/processing_thread.py
from PyQt5.QtCore import QThread
from src.signals import WorkerSignals

class ProcessingThread(QThread):
    def __init__(self, parent, func, signals, *args, **kwargs):
        super().__init__(parent)
        self.func = func
        self.signals = signals
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.func(*self.args, **self.kwargs)

    def stop(self):
        self.signals._stopped = True  # Устанавливаем флаг напрямую в signals
        self.terminate()  # Прерываем цикл событий потока
        self.wait()  # Ждём завершения