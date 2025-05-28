import os
import matplotlib.pyplot as plt
from PIL import Image
# Предполагается, что DatasetLoader и ImageGenerator находятся в этих модулях
from dataset_loader import DatasetLoader
from image_generator import ImageGenerator


def show_images_from_dir(directory, num_images=3):
    """
    Отображает первые N изображений из указанной директории.
    
    Параметры:
        directory (str): Путь к папке с изображениями.
        num_images (int): Количество отображаемых изображений.
    """
    if not os.path.exists(directory):
        print(f"Ошибка: Директория '{directory}' не найдена.")
        return

    image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:num_images]

    if not image_files:
        print("Нет подходящих изображений для отображения.")
        return

    plt.figure(figsize=(15, 5))
    for idx, image_file in enumerate(image_files):
        img_path = os.path.join(directory, image_file)
        img = Image.open(img_path)
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{image_file}")
    plt.tight_layout()
    plt.show()

# Запуск через ./task2 !!
if __name__ == "__main__":

    # Указываем пути
    dataset_path = "MyDataset"
    save_path = "Results"

    # Удаляем старый результат
    # shutil.rmtree(save_path)

    # Создаём загрузчик датасета
    dataset_loader = DatasetLoader(dataset_path)

    # Инициализируем генератор
    generator = ImageGenerator(
        dataset_loader,
        img_size=(480, 640),
        num_imgs=2,
        num_cells=69,
        seed=69
    )

    # Генерируем и сохраняем изображения
    generator.generate_and_save(save_path)

    # Отображаем одно из сгенерированных изображений
    show_images_from_dir(save_path, num_images=3)