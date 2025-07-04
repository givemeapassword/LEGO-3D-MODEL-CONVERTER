# LEGO-Compatible Cubic Assembly Generator 🚀

Добро пожаловать в проект, который превращает ваши 3D-модели в LEGO-подобные сборки!  
Система автоматически разбивает модель на кубические элементы, оптимизирует конструкцию и экспортирует её для 3D-печати или ручной сборки.

---

## ✨ Возможности

- **Импорт 3D-моделей** 📥  
  Поддержка форматов `.STL`, `.OBJ`, `.3MF`. Автоматическая проверка и исправление геометрии.

- **Адаптивная вокселизация** 🧮  
  Умный алгоритм разбивки на кубы с учётом формы и поверхности модели.

- **Оптимизация сборки** ⚙️  
  Меньше деталей — больше прочности. Алгоритмы минимизируют количество блоков без потери устойчивости.

- **Интуитивный интерфейс** 🖼️  
  Визуализация процесса и настройка параметров через PyQt5 GUI.

- **Экспорт результатов** 📄  
  - STL-файлы для 3D-печати  
  - PDF-инструкции с пошаговыми иллюстрациями

---

## 🛠️ Технологии

- **Язык программирования:** Python 🐍  
- **Библиотеки и инструменты:**
  - `Trimesh` — обработка 3D-моделей
  - `Numba` — ускорение вокселизации
  - `PyQt5` — графический интерфейс
  - `PyVista` — 3D-визуализация
  - `ReportLab`, `Pillow` — генерация PDF-инструкций
  - `Scikit-learn`, `NumPy` — алгоритмы оптимизации
- **IDE:** Visual Studio Code 💻

---

## 🚧 Ограничения

- Поддержка моделей до 1 млн полигонов
- Используются только стандартные кубические LEGO-элементы
- Сложные мелкие детали упрощаются для ускорения обработки

---

## 🌟 Почему это круто?

Система — мост между цифровым дизайном и физическим творчеством.  
Подходит для:

- **Образования** 📚 — изучение геометрии, робототехники, 3D-дизайна  
- **DIY-энтузиастов** 🔧 — уникальные модели без сложных инструментов  
- **Прототипирования** 🛠️ — быстрые физические макеты

Преобразуйте 3D-модели в LEGO-шедевры всего за пару минут! ⏱️

---

## 🤝 Вклад в проект

Хотите улучшить проект? Мы открыты к вашим идеям! 💡 Форкните репозиторий, создайте pull request или напишите нам об ошибках и предложениях.

## 📜 Лицензия

Проект распространяется под лицензией MIT. Используйте, модифицируйте и делитесь свободно! 😊

---

Создавайте, стройте, вдохновляйте! 🧱
