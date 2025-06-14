import logging
from collections import deque
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QTextCharFormat, QColor, QFont
from src.config.config import LOG_UPDATE_INTERVAL

class QTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str, str)

    def __init__(self, text_edit):
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self.text_edit = text_edit
        self.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.log_signal.connect(self.append_text)
        self.log_queue = deque(maxlen=10)
        self.last_update = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.flush_logs)
        self.timer.start(LOG_UPDATE_INTERVAL)

    def emit(self, record):
            if record.levelno >= self.level:
                msg = self.format(record)
                level = record.levelname
                self.log_queue.append((msg, level))

    def flush_logs(self):
        import time
        current_time = time.time()
        if current_time - self.last_update >= 0.5:
            while self.log_queue:
                msg, level = self.log_queue.popleft()
                self.log_signal.emit(msg, level)
            self.last_update = current_time

    def append_text(self, msg, level):
        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.End)
        self.text_edit.setTextCursor(cursor)
        format = QTextCharFormat()
        format.setFontFamily('Poppins')  # Используем ваш шрифт
        format.setFontPointSize(12)

        # Добавляем иконки и цвета
        if "Processing" in msg:
            format.setForeground(QColor("#f05d22"))
            format.setFontWeight(QFont.Bold)
            self.text_edit.insertPlainText("⚙️ ")  # Иконка шестерёнки
        elif level == "INFO":
            format.setForeground(QColor("#4c4c4c"))
            self.text_edit.insertPlainText("ℹ️ ")  # Иконка информации
        elif level == "WARNING":
            format.setForeground(QColor("#FFA500"))
            self.text_edit.insertPlainText("⚠️ ")  # Иконка предупреждения
        elif level == "ERROR":
            format.setForeground(QColor("#FF4040"))
            self.text_edit.insertPlainText("❌ ")  # Иконка ошибки
        else:
            format.setForeground(QColor("#666666"))
            self.text_edit.insertPlainText("➡️ ")  # Иконка по умолчанию

        self.text_edit.setCurrentCharFormat(format)
        self.text_edit.insertPlainText(msg + "\n")
        self.text_edit.ensureCursorVisible()