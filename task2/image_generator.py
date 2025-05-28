import os
import cv2
import random
import numpy as np
from dataset_loader import DatasetLoader


class ImageGenerator:
    def __init__(self, dataset_loader, img_size=(480, 640), num_imgs=5, num_cells=25, seed=None):
        """
        Генератор изображений с наложенными кровяными клетками.

        Параметры:
            dataset_loader (DatasetLoader): Загрузчик данных.
            img_size (tuple): Размер выходного изображения (высота, ширина).
            num_imgs (int): Количество генерируемых изображений.
            num_cells (int): Количество клеток на каждом изображении.
            seed (int or None): Сид для воспроизводимости случайных чисел.
        """
        self.dataset_loader = dataset_loader
        self.img_size = img_size
        self.num_imgs = num_imgs
        self.num_cells = num_cells

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_and_save(self, save_dir):
        """
        Генерирует заданное количество изображений и сохраняет их в указанной директории.

        Параметры:
            save_dir (str): Путь к папке для сохранения изображений.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(self.num_imgs):
            image = self._generate_image()
            filename = os.path.join(save_dir, f"result_{idx}.png")
            cv2.imwrite(filename, image)

    def _generate_image(self):
        """Генерирует одно изображение с несколькими случайными клетками."""
        background_path = self.dataset_loader.get_random_background()
        background = self._load_image(background_path, resize=True)

        coords = self._generate_cell_coords()

        for i in range(self.num_cells):
            cell_path = self.dataset_loader.get_random_cell()
            cell = self._load_image(cell_path, resize=True, with_alpha=True)

            transparency = random.uniform(0.6, 1.0)
            background = self._overlay(background, cell, coords[i], transparency)

        return background

    def _generate_cell_coords(self):
        """Генерирует случайные координаты для размещения клеток."""
        return [
            {
                "h": random.randint(0, self.img_size[0] - int(self.img_size[0] * 0.01)),
                "w": random.randint(0, self.img_size[1] - int(self.img_size[1] * 0.01))
            }
            for _ in range(self.num_cells)
        ]

    def _load_image(self, image_path, resize=False, with_alpha=False):
        """
        Загружает изображение.

        Параметры:
            image_path (str): Путь к файлу.
            resize (bool): Изменять ли размер под целевой размер.
            with_alpha (bool): Загружать ли с альфа-каналом.
        """
        flags = cv2.IMREAD_UNCHANGED if with_alpha else cv2.IMREAD_COLOR
        image = cv2.imread(image_path, flags)

        if image is None:
            raise ValueError(f"Ошибка: Не удалось загрузить изображение '{image_path}'")

        if resize:
            target_size = (self.img_size[1], self.img_size[0])  # (ширина, высота)
            image = cv2.resize(image, target_size)

        return image

    def _overlay(self, background, cell, coord, transparency=1.0):
        """
        Накладывает клетку на фоновое изображение.

        Параметры:
            background (np.array): Фоновое изображение.
            cell (np.array): Клетка с альфа-каналом.
            coord (dict): Координаты {'h': y, 'w': x}.
            transparency (float): Прозрачность наложения (0.0–1.0).

        Возвращает:
            np.array: Обновлённое фоновое изображение.
        """
        scale = round(random.uniform(0.05, 0.20), 2)
        cell_size = int(min(self.img_size) * scale)
        cell = cv2.resize(cell, (cell_size, cell_size))

        cell = self._rotate_image(cell)

        h, w = coord["h"], coord["w"]
        cell_h, cell_w = cell.shape[:2]

        # Ограничиваем размеры, чтобы не выйти за границы
        cell_h = min(cell_h, background.shape[0] - h)
        cell_w = min(cell_w, background.shape[1] - w)
        cell = cell[:cell_h, :cell_w]

        if cell.shape[2] == 4:
            alpha = (cell[:, :, 3] / 255.0) * transparency
            for c in range(3):
                background[h:h + cell_h, w:w + cell_w, c] = (
                    (1 - alpha) * background[h:h + cell_h, w:w + cell_w, c] +
                    alpha * cell[:, :, c]
                )

        return background

    def _rotate_image(self, image):
        """
        Поворачивает изображение на случайный угол и сохраняет его пропорции.

        Параметры:
            image (np.array): Изображение с альфа-каналом.

        Возвращает:
            np.array: Повёрнутый массив изображения.
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        angle = random.randint(0, 360)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Рассчитываем новый размер изображения после поворота
        new_w = int((height * abs(matrix[0, 1])) + (width * abs(matrix[0, 0])))
        new_h = int((height * abs(matrix[0, 0])) + (width * abs(matrix[0, 1])))

        # Центрируем изображение после поворота
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return rotated