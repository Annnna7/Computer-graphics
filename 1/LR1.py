from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt  
import numpy as np
import re
import math
import cv2

# 1.1
# "создаем матрицу нулей"
# matrix = np.zeros((100, 200), dtype=np.uint8)
# "создаем изображение из матрицы"
# image = Image.fromarray(matrix, mode='L')
# "сохраняем изображение в файл"
# image.save('black_image.png')
# image.show()

# 1.2
# matrix = np.full((100, 200), 255, dtype=np.uint8)
# image = Image.fromarray(matrix, mode='L')
# image.save('white_image.png')
# image.show()

# 1.3
# matrix = np.full((100, 200, 3), (255, 0, 0), dtype=np.uint8)
# image = Image.fromarray(matrix)
# image.save('image.png')
# image.show()

# 1.4
# matrix = np.full((60, 200, 3), (255, 0, 0), dtype=np.uint8)
# for i in range(60):
#     for j in range(200):
#         matrix[i, j] = (i+j)%256
# image = Image.fromarray(matrix)
# image.save('image.png')
# image.show()

# 2.1
# рисовать пиксели с заданным шагом, интерполируя x и y между начальным и конечным значениями
# def dotted_line(image, x0, y0, x1, y1, count, color):
#     step = 1.0/count
#     for t in np.arange(0, 1, step):
#         x = round((1.0 - t) * x0 + t * x1)
#         y = round((1.0 - t) * y0 + t * y1)
#         image[y, x] = color
        
# 2.2
# Можно выбирать шаг на основе расстояния между первой и последней точкой
# def dotted_line(image, x0, y0, x1, y1,color):
#     count = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
#     step = 1.0/count
#     for t in np.arange (0, 1, step):
#         x = round ((1.0 - t)*x0 + t*x1)
#         y = round ((1.0 - t)*y0 + t*y1)
#         image[y, x] = color
        
# 2.3
# def dotted_line(image, x0, y0, x1, y1, color):
#     steep = False
#     if abs(x0 - x1) < abs(y0 - y1):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         steep = True
#     if x0 > x1:
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#     
#         
#     for x in range (int(x0), int(x1)):
#         t = (x - x0)/(x1 - x0)
#         y = round((1.0 - t)*y0 + t*y1)
#         if (steep):
#             image[int(x), int(y)] = color
#         else:
#             image[int(y), int(x)] = color
            
# 2.4 алгоритм  Брезенхема
def dotted_line(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = abs(y1 - y0) * 2
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(int(x0), int(x1)):
        if xchange:
            image[int(x), int(y)] = color
        else:
            image[int(y), int(x)] = color
        derror += dy
        if (derror >= x1 - x0):
            derror -= 2 * (x1 - x0)
            y += y_update

# далее +- общее для 2ого задания
# Создание тестового изображения
# image = np.zeros((200, 200, 3), dtype=np.uint8)
# # Рисование прямой линии
# for i in range (0, 13):
#     a=2*3.1416*i/13
#     x1=100+95*math.cos(a)
#     y1=100+95*math.sin(a)
#     # для 2.1 используем 
# #     dotted_line(image, 100, 100, x1, y1, 100, (0, 0, 255))  
#     # для 2.2 - 2.4 используем
#     dotted_line(image, 100, 100, x1, y1, (0, 0, 255))  
# # Визуализация результата
# cv2.imshow('Dotted Line Test', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 3
# class Vertex:
#     def __init__(self, x, y, z):
#         self.x = x
#         self.y = y
#         self.z = z
# vertices = []
# with open("model_1.obj", "r") as file:
#     for line in file:
#         if line.startswith("v "):
#             data = line.split()
# #преобразование элементов строки в числа с плавающей запятой
#             x, y, z = map(float, data[1:])
#             vertices.append(Vertex(x, y, z))
# 
# for vertex in vertices:
#     print(f"Vertex: ({vertex.x}, {vertex.y}, {vertex.z})")

# 4
class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
vertices = []
with open("model_2.obj", "r") as file:
    for line in file:
        if line.startswith("v "):
            data = line.split()
            x, y, z = map(float, data[1:])
            vertices.append(Vertex(x, y, z))

# # создание новой фигуры для построения графика с размером 10x10 дюймов
# plt.figure(figsize=(10, 10))
# scaled_x = []
# scaled_y = []
# for vertex in vertices:
#     scaled_x.append(50*vertex.x+500)
#     scaled_y.append(50*vertex.y+500)
# #     построение точечной диаграммы с масштабированными координатами x и y в виде точек с цветом синий
# plt.scatter(scaled_x, scaled_y, color='blue')
# plt.show()

# 5
# Функция для чтения файла с данными о полигонах
def read_model_data(file_name):
    polygons = []
# используется для безопасного открытия файла для чтения и гарантирует его автоматическое закрытие после использования.
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith('f'):
                vertices = line.split()[1:]  # Получаем список вершин для каждого полигона
                polygon_vertices = [int(vertex.split('/')[0]) for vertex in vertices]  # Берем только номера вершин
                polygons.append(polygon_vertices)
    return polygons

# Загрузка данных о полигонах из файла
# polygons_data = read_model_data('model_1.obj')
# 
# # Вывод данных о полигонах
# for i, polygon in enumerate(polygons_data, start=1):
#     print(f'Полигон {i}: {polygon}')
    
# 6
polygons = read_model_data('model_2.obj')           
# rabbit = np.zeros((2000, 2000), dtype=np.uint8)
deer = np.zeros((2000, 2000), dtype=np.uint8)
for i in range(0, len(polygons)):
    p1 = polygons[i][0] - 1
    p2 = polygons[i][1] - 1
    p3 = polygons[i][2] - 1
#     x1 = int(vertices[p1].x * 10000) + 1000
#     y1 = int(vertices[p1].y * 10000) + 1000
#     x2 = int(vertices[p2].x * 10000) + 1000
#     y2 = int(vertices[p2].y * 10000) + 1000
#     x3 = int(vertices[p3].x * 10000) + 1000
#     y3 = int(vertices[p3].y * 10000) + 1000

    x1 = int(vertices[p1].x * 0.5) + 1000
    y1 = int(vertices[p1].y * 0.5) + 1000
    x2 = int(vertices[p2].x * 0.5) + 1000
    y2 = int(vertices[p2].y * 0.5) + 1000
    x3 = int(vertices[p3].x * 0.5) + 1000
    y3 = int(vertices[p3].y * 0.5) + 1000
#     dotted_line(rabbit, int(x1), int(y1), int(x2), int(y2), 255)
#     dotted_line(rabbit, int(x2), int(y2), int(x3), int(y3), 255)
#     dotted_line(rabbit, int(x3), int(y3), int(x1), int(y1), 255)

    dotted_line(deer, x1, y1, x2, y2, 255)
    dotted_line(deer, x2, y2, x3, y3, 255)
    dotted_line(deer, x3, y3, x1, y1, 255)

# img = Image.fromarray(rabbit, mode='L')
img = Image.fromarray(deer, mode='L')
img = ImageOps.flip(img)
# img.save('rabbit.png')
img.save('deer.png')
img.show()

    
    












