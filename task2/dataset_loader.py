import os
import random

class DatasetLoader:
    def __init__(self, dataset_dir):
        """
        Загружает пути к изображениям из подпапок 'Backgrounds' и 'BloodCells'.
        
        Параметры:
            dataset_dir (str): Путь к основной папке датасета.
        """
        self.dataset_dir = dataset_dir
        self.patterns = {
            "Backgrounds": [],
            "BloodCells": []
        }
        self._load_patterns()

    def _load_patterns(self):
        """Сканирует папку датасета и собирает пути к файлам в нужных подкаталогах."""
        for root, dirs, files in os.walk(self.dataset_dir):
            category = os.path.basename(root)

            if category in self.patterns:
                # Сохраняем отсортированные пути к файлам
                image_paths = [os.path.join(root, file) for file in files]
                self.patterns[category] = sorted(image_paths)

        # Проверяем, что обе папки найдены и содержат файлы
        if not self.patterns["Backgrounds"]:
            raise ValueError("Не найдено изображений в папке: Backgrounds")
        
        if not self.patterns["BloodCells"]:
            raise ValueError("Не найдено изображений в папке: BloodCells")

    def get_random_background(self):
        """Возвращает случайный путь к фоновому изображению."""
        return random.choice(self.patterns["Backgrounds"])

    def get_random_cell(self):
        """Возвращает случайный путь к изображению кровяной клетки."""
        return random.choice(self.patterns["BloodCells"])