import numpy as np
import cv2

class ImageProcessor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def additive_noise(image, percent):

        return np.clip(np.where(np.random.rand(*image.shape) < percent / 100,
                        image + np.random.randint(-20, 20, image.shape), image), 0, 255)

    @staticmethod
    def mean_filter(image, kernel_size):

        return cv2.blur(image, (kernel_size, kernel_size))

    
    @staticmethod
    def gauss_filter(image, kernel_size):
        sigma = kernel_size // 2 / 3

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
       
    
    @staticmethod
    def image_equalization(image):

 
        return cv2.equalizeHist(image)
    

    @staticmethod
    def statistic_correction(image, new_mean, new_std):
        image = image.astype(np.float32)
        mean, std = image.mean(), image.std()
        
        corrected_image = (image - mean) * (new_std / std) + new_mean
        corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
        
        return corrected_image


    @staticmethod
    def resize(image, new_width, new_height):
        

 
        height, width = image.shape
        y_indices = (np.arange(new_height) * height / new_height).astype(int)
        x_indices = (np.arange(new_width) * width / new_width).astype(int)

        resized = image[np.ix_(y_indices, x_indices)]
        return resized


    @staticmethod
    def shift(image, x, y):
        
        h, w = image.shape
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return shifted

    
    @staticmethod
    def rotation(image, k, l, angle):
        
        height, width = image.shape[:2]

        rotation_matrix = cv2.getRotationMatrix2D((k, l), -angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated_image


    @staticmethod
    def glass_effect(image):
        
        height, width = image.shape[:2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        rand_x = np.clip(x + np.random.randint(-5, 6, x.shape), 0, width - 1)
        rand_y = np.clip(y + np.random.randint(-5, 6, y.shape), 0, height - 1)

        return image[rand_y, rand_x]
    

    @staticmethod
    def waves(image):

        height, width = image.shape[:2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        rand_x = np.clip(x + 20 * np.sin(2 * np.pi * y / 30), 0, width - 1).astype(int)

        return image[y, rand_x]
    
    @staticmethod
    def motion_blur(image, n):
        
        kernel = np.zeros((n, n))
        kernel[np.arange(n), np.arange(n)] = 1
        kernel /= n

        return cv2.filter2D(image, -1, kernel)

