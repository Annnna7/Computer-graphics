from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt  
import numpy as np
import re, random
import math

class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
vertices = []
with open("model2.obj", "r") as file:
    for line in file:
        if line.startswith("v "):
            data = line.split()
            x, y, z = map(float, data[1:])
            vertices.append(Vertex(x, y, z))
            
def read_model_data(file_name):
    polygons = []
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith('f'):
                vertices = line.split()[1:]  
                polygon_vertices = [int(vertex.split('/')[0]) for vertex in vertices]  
                polygons.append(polygon_vertices)
    return polygons
    
# 7
polygons = read_model_data('model2.obj')     
def bar_cord(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

#8-9
#Отрисовка треугольников
def draw_a_triangle(img, x0, y0, x1, y1, x2, y2, color):
    xmin = int(min(x0, x1, x2))
    ymin = int(min(y0, y1, y2))
    xmax = int(max(x0, x1, x2))
    ymax = int(max(y0, y1, y2))
    # Учтем границы изображения
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            # Вычисляем барицентрические координаты
            lambda0, lambda1, lambda2 = bar_cord(x, y, x0, y0, x1, y1, x2, y2)
            # Проверяем условие барицентрических координат
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                img[int(y), int(x)] = color

image = np.zeros((1000, 1000, 3), dtype=np.uint8)
image2 = np.zeros((1000, 1000, 3), dtype=np.uint8)
draw_a_triangle(image, 57, 356, 20, 750, 120, 100, (0, 128, 0))
img = Image.fromarray(image, mode='RGB')
img.save('new1.png')
img.show()

#10
#Отрисовка полигонов трёхмерной модели
image = np.zeros((1000, 1000, 3), dtype=np.uint8)
for i in range(0, len(polygons)):
    p1 = polygons[i][0] - 1
    p2 = polygons[i][1] - 1
    p3 = polygons[i][2] - 1
    x0 = vertices[p1].x * 5000 + 500
    y0 = vertices[p1].y * 5000 + 500
    x1 = vertices[p2].x * 5000 + 500
    y1 = vertices[p2].y * 5000 + 500
    x2 = vertices[p3].x * 5000 + 500
    y2 = vertices[p3].y * 5000 + 500
#     x0 = vertices[p1].x * 0.3 + 500
#     y0 = vertices[p1].y * 0.3 + 500
#     x1 = vertices[p2].x * 0.3 + 500
#     y1 = vertices[p2].y * 0.3 + 500
#     x2 = vertices[p3].x * 0.3 + 500
#     y2 = vertices[p3].y * 0.3 + 500
    draw_a_triangle(image, x0, y0, x1, y1, x2, y2, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
img = Image.fromarray(image, mode='RGB')
img = ImageOps.flip(img)
img.save('new1.png')
img.show()

# 12
def def_cos(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    l = [0, 0, 1]
    # 11
    normal = [(y1 - y2) * (z1 - z0) - (z1 - z2) * (y1 - y0),
              (z1 - z2) * (x1 - x0) - (x1 - x2) * (z1 - z0),
              (x1 - x2) * (y1 - y0) - (y1 - y2) * (x1 - x0)]
    composition = normal[0]*l[0] + normal[1]*l[1] + normal[2]*l[2]
    norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    ll = np.sqrt(l[0]**2 + l[1]**2 + l[2]**2)
    cos = composition/(norm*ll)
    return cos

# 13
def draw_a_triangle(z_buf, img, x0, y0, z0, x1, y1, z1, x2, y2, z2, color):
    xmin = int(min(x0, x1, x2))
    ymin = int(min(y0, y1, y2))
    xmax = int(max(x0, x1, x2))
    ymax = int(max(y0, y1, y2))
    # Учтем границы изображения
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            # Вычисляем барицентрические координаты
            l0, l1, l2 = bar_cord(x, y, x0, y0, x1, y1, x2, y2)
            # Проверяем условие барицентрических координат
            if l0 >= 0 and l1 >= 0 and l2 >= 0 and (def_cos(x0, y0, z0, x1, y1, z1, x2, y2, z2) < 0):
                # 14
                z = l0  * z0 + l1 * z1 + l2 * z2
                if z < z_buf[y][x]:
                    img[y][x] = color
                    z_buf[y][x] = z

image2 = np.zeros((1000, 1000, 3), dtype=np.uint8)
z_buff = np.full((1000, 1000), np.inf, dtype=np.float32)
for i in range(0, len(polygons)):
    p1 = polygons[i][0] - 1
    p2 = polygons[i][1] - 1
    p3 = polygons[i][2] - 1
    x0 = vertices[p1].x * 5000 + 500
    y0 = vertices[p1].y * 5000 + 500
    z0 = vertices[p1].z * 5000 + 500
    x1 = vertices[p2].x * 5000 + 500
    y1 = vertices[p2].y * 5000 + 500
    z1 = vertices[p2].z * 5000 + 500
    x2 = vertices[p3].x * 5000 + 500
    y2 = vertices[p3].y * 5000 + 500
    z2 = vertices[p3].z * 5000 + 500
    draw_a_triangle(z_buff, image2, x0, y0, z0, x1, y1, z1, x2, y2, z2,
    [-255*def_cos(x0, y0, z0, x1, y1, z1, x2, y2, z2), 0, 0])
img = Image.fromarray(image2, mode='RGB')
img = ImageOps.flip(img)
img.save('img2.png')
img.show()


