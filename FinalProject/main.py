from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from image_processor import ImageProcessor
from augmentator import Augmentator
import shutil
import random

augmentator = Augmentator()



from enum import Enum
class Command(Enum):
    ADDITIVE_NOISE = "Зашумление"
    MEAN_FILTER = "Фильтр среднего"
    GAUSS_FILTER = "Фильтр Гаусса"
    EQUALIZATION = "Эквализация"
    STATISTIC_CORRECTION = "Статистическая цветокоррекция"
    RESIZE = "Масштабирование"
    SHIFT = "Перенос"
    ROTATION = "Поворот"
    GLASS_EFFECT = "Эффект стекла"
    WAVES = "Волны"
    MOTION_BLUR = "Motion Blur"




def set_image_to_label(label, image, zoom=1):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(
        image.resize((int(image.width * zoom), int(image.height * zoom)))
    )
    label.configure(image=image)
    label.image = image


def on_load_image_button_click():
    path = filedialog.askdirectory(title='Выберите директорию с изображениями')

    if path:
        
        augmentator.load_images(path, int(augment_percent_entry.get() or '0'))
        

        if augmentator.opened_files:
            set_image_to_label(image_label, augmentator.opened_files[0])


def on_do_image_button_click():

    selection = augmentation_algorithm_combo.get()
    

    for i, loaded_image in enumerate(augmentator.opened_files or []):

        
        match selection:
            case Command.ADDITIVE_NOISE.value:
                percent = int(noise_entry.get())
                loaded_image = ImageProcessor.additive_noise(loaded_image, percent)
            case Command.MEAN_FILTER.value:
                power = int(denoise_entry.get())
                loaded_image = ImageProcessor.mean_filter(loaded_image, power)
            case Command.GAUSS_FILTER.value:
                power = int(denoise_entry.get())
                loaded_image = ImageProcessor.gauss_filter(loaded_image, power)
            case Command.EQUALIZATION.value:
                loaded_image = ImageProcessor.image_equalization(loaded_image)
            case Command.STATISTIC_CORRECTION.value:
                mean = float(statistic_correction_mean_entry.get())
                std = float(statistic_correction_std_entry.get())
                loaded_image = ImageProcessor.statistic_correction(loaded_image, mean, std)
            case Command.RESIZE.value:
                width = int(scale_w_entry.get())
                height = int(scale_h_entry.get())
                loaded_image = ImageProcessor.resize(loaded_image, width, height)
            case Command.SHIFT.value:
                x = int(translation_x_entry.get())
                y = int(translation_y_entry.get())
                loaded_image = ImageProcessor.shift(loaded_image, x, y)
            case Command.ROTATION.value:
                x = int(rotation_x_entry.get())
                y = int(rotation_y_entry.get())
                angle = int(rotation_angle_entry.get())
                loaded_image = ImageProcessor.rotation(loaded_image, angle)
            case Command.GLASS_EFFECT.value:
                loaded_image = ImageProcessor.glass_effect(loaded_image)
            case Command.WAVES.value:
                loaded_image = ImageProcessor.waves(loaded_image)
            case Command.MOTION_BLUR.value:
                n = int(motion_blur_power_entry.get())
                loaded_image = ImageProcessor.motion_blur(loaded_image, n)
            case _:
                pass

        augmentator.opened_files[i] = loaded_image

    if augmentator.opened_files:
        set_image_to_label(image_label, augmentator.opened_files[0])


def on_save_image_button_click():
    if augmentator.opened_files is None:
        return
    
    path = filedialog.askdirectory(title="Выбрать папку для сохранения")

    if path:
        shutil.rmtree(path, True)
        shutil.copytree(augmentator.base_dir, path)

        for label, img in zip(augmentator.opened_labels, augmentator.opened_files):
            names = label[1:].split('\\')
            cv2.imwrite(os.path.join(path, names[0], 'aug_' + names[1]), img)



def on_augmentation_algorithm_combo_selected(event):
    noise_frame.grid_forget()
    denoise_frame.grid_forget()
    denoise_gauss_frame.grid_forget()
    scale_frame.grid_forget()
    statistic_correction_frame.grid_forget()
    translation_frame.grid_forget()
    rotation_frame.grid_forget()
    motion_blur_frame.grid_forget()

    selection = augmentation_algorithm_combo.get()

    match selection:
        case Command.ADDITIVE_NOISE.value:
            noise_frame.grid(row=2, column=0, sticky=EW, padx=5, pady=5)
        case Command.MEAN_FILTER.value:
            denoise_frame.grid(row=2, column=0, sticky=EW, padx=5, pady=5)
        case Command.GAUSS_FILTER.value:
            denoise_gauss_frame.grid(row=2, column=0, sticky=EW, padx=5, pady=5)
        case Command.RESIZE.value:
            scale_frame.grid(row=2, column=0, sticky=EW, padx=5, pady=5)
        case Command.STATISTIC_CORRECTION.value:
            statistic_correction_frame.grid(row=2, column=0, sticky=EW, padx=5, pady=5)
        case Command.SHIFT.value:
            translation_frame.grid(row=2, column=0, sticky=EW, padx=5, pady=5)
        case Command.ROTATION.value:
            rotation_frame.grid(row=2, column=0, sticky=EW, padx=5, pady=5)
        case Command.MOTION_BLUR.value:
            motion_blur_frame.grid(row=2, column=0, sticky=EW, padx=5, pady=5)
    

window = Tk()
window.minsize(800, 600)
window.title("Аугментация данных")

window.columnconfigure(0, weight=1)

file_operations_frame = Frame(window)
file_operations_frame.grid(row=0, column=0, sticky=EW, padx=5, pady=5)

load_image_button = ttk.Button(
    file_operations_frame,
    text="Загрузить изображение",
    command=on_load_image_button_click,
)
load_image_button.pack(side=TOP)
do_image_button = ttk.Button(
    file_operations_frame,
    text="Преобразовать изображение",
    command=on_do_image_button_click,
)
do_image_button.pack(side=TOP)
save_image_button = ttk.Button(
    file_operations_frame, 
    text="Сохранить изображение",
    command=on_save_image_button_click,)
save_image_button.pack(side=TOP)



base_settings_frame = Frame(window)


augment_percent_label = ttk.Label(base_settings_frame, text="Процент аугментации")
augment_percent_label.pack(side=LEFT)
augment_percent_entry = ttk.Entry(base_settings_frame)
augment_percent_entry.pack(fill=X, side=LEFT, expand=True)
augment_percent_entry.insert(0, "10")

base_settings_frame.grid(row=1, column=0, sticky=EW, padx=5, pady=5)
augmentation_algorithm_label = ttk.Label(
    base_settings_frame, text="Алгоритм преобразования"
)
augmentation_algorithm_label.pack(side=LEFT)
augmentation_algorithm_combo = ttk.Combobox(
    base_settings_frame,



    values=[
        Command.ADDITIVE_NOISE.value,
        Command.MEAN_FILTER.value,
        Command.GAUSS_FILTER.value,
        Command.EQUALIZATION.value,
        Command.STATISTIC_CORRECTION.value,
        Command.RESIZE.value,
        Command.SHIFT.value,
        Command.ROTATION.value,
        Command.GLASS_EFFECT.value,
        Command.WAVES.value,
        Command.MOTION_BLUR.value,
    ],
   

)
augmentation_algorithm_combo.bind("<<ComboboxSelected>>", on_augmentation_algorithm_combo_selected)
augmentation_algorithm_combo.pack(fill=X, side=LEFT, expand=True)

window.rowconfigure(3, weight=1)
canvas = Canvas(bg="white")
canvas.grid(row=3, column=0, sticky=NSEW, padx=5, pady=5)
image_label = Label(canvas)
image_label.place(relx=0.5, rely=0.5, anchor="center")

noise_frame = Frame(window)
noise_label = ttk.Label(
    noise_frame, text="Процент шумных пикселей"
)
noise_label.pack(side=LEFT)
noise_entry = ttk.Entry(noise_frame)
noise_entry.pack(fill=X, side=LEFT, expand=True)
noise_entry.insert(0, "30")

denoise_frame = Frame(window)
denoise_label = ttk.Label(denoise_frame, text="Сила шумоподавления")
denoise_label.pack(side=LEFT)
denoise_entry = ttk.Entry(denoise_frame)
denoise_entry.pack(fill=X, side=LEFT, expand=True)
denoise_entry.insert(0, "5")

denoise_gauss_frame = Frame(window)
denoise_gauss_label = ttk.Label(denoise_gauss_frame, text="Сила шумоподавления")
denoise_gauss_label.pack(side=LEFT)
denoise_gauss_entry = ttk.Entry(denoise_gauss_frame)
denoise_gauss_entry.pack(fill=X, side=LEFT, expand=True)
denoise_gauss_entry.insert(0, "5")

scale_frame = Frame(window)
scale_w_label = ttk.Label(scale_frame, text="Новая ширина")
scale_h_label = ttk.Label(scale_frame, text="Новая высота")
scale_w_entry = ttk.Entry(scale_frame)
scale_h_entry = ttk.Entry(scale_frame)
scale_frame.columnconfigure(1, weight=1)
scale_w_label.grid(row=0, column=0, sticky=W)
scale_w_entry.grid(row=0, column=1, sticky=EW)
scale_h_label.grid(row=1, column=0, sticky=W)
scale_h_entry.grid(row=1, column=1, sticky=EW)
scale_w_entry.insert(0, "100")
scale_h_entry.insert(0, "100")

translation_frame = Frame(window)
translation_x_label = ttk.Label(translation_frame, text="Перенос по X")
translation_y_label = ttk.Label(translation_frame, text="Перенос по Y")
translation_x_entry = ttk.Entry(translation_frame)
translation_y_entry = ttk.Entry(translation_frame)
translation_frame.columnconfigure(1, weight=1)
translation_x_label.grid(row=0, column=0, sticky=W)
translation_x_entry.grid(row=0, column=1, sticky=EW)
translation_y_label.grid(row=1, column=0, sticky=W)
translation_y_entry.grid(row=1, column=1, sticky=EW)
translation_x_entry.insert(0, "50")
translation_y_entry.insert(0, "50")



statistic_correction_frame = Frame(window)

statistic_correction_mean_label = ttk.Label(statistic_correction_frame, text="Среднее")
statistic_correction_std_label = ttk.Label(statistic_correction_frame, text="Дисперсия")
statistic_correction_mean_entry = ttk.Entry(statistic_correction_frame)
statistic_correction_std_entry = ttk.Entry(statistic_correction_frame)


statistic_correction_frame.columnconfigure(1, weight=1)

statistic_correction_mean_label.grid(row=0, column=0, sticky=W)
statistic_correction_mean_entry.grid(row=0, column=1, sticky=EW)
statistic_correction_std_label.grid(row=1, column=0, sticky=W)
statistic_correction_std_entry.grid(row=1, column=1, sticky=EW)
statistic_correction_mean_entry.insert(0, "100")
statistic_correction_std_entry.insert(0, "50")



rotation_frame = Frame(window)
rotation_label = ttk.Label(rotation_frame, text="Угол поворота")
rotation_label.pack(side=LEFT)
rotation_x_entry = ttk.Entry(rotation_frame)
rotation_x_entry.pack(fill=X, side=LEFT, expand=True)
rotation_y_entry = ttk.Entry(rotation_frame)
rotation_y_entry.pack(fill=X, side=LEFT, expand=True)
rotation_angle_entry = ttk.Entry(rotation_frame)
rotation_angle_entry.pack(fill=X, side=LEFT, expand=True)
rotation_x_entry.insert(0, "50")
rotation_y_entry.insert(0, "50")
rotation_angle_entry.insert(0, "30")

motion_blur_frame = Frame(window)
motion_blur_power_label = ttk.Label(motion_blur_frame, text="Сила эффекта")
motion_blur_power_entry = ttk.Entry(motion_blur_frame)
motion_blur_frame.columnconfigure(1, weight=1)
motion_blur_power_label.grid(row=0, column=0, sticky=W)
motion_blur_power_entry.grid(row=0, column=1, sticky=EW)
motion_blur_power_entry.insert(0, "10")


window.mainloop()
