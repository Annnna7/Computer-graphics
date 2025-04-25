import random
import numpy as np
from PIL import Image, ImageOps

class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def bar(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def turn(a, b, c, t, old):
    r1 = np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])
    r2 = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    r3 = np.array([[np.cos(c), np.sin(c), 0], [-np.sin(c), np.cos(c), 0], [0, 0, 1]])
    result = np.dot(np.dot(np.dot(r1, r2), r3), np.array(old)) + np.array(t)
    return result

def draw(z_buf, img, x0, y0, z0, x1, y1, z1, x2, y2, z2, color):
    p_x0 = 20*x0/z0 + 500
    p_y0 = 20*y0/z0 + 500
    p_x1 = 20*x1/z1 + 500
    p_y1 = 20*y1/z1 + 500
    p_x2 = 20*x2/z2 + 500
    p_y2 = 20*y2/z2 + 500

    xmin = int(min(p_x0, p_x1, p_x2)) 
    ymin = int(min(p_y0, p_y1, p_y2)) 
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    xmax = int(max(p_x0, p_x1, p_x2) + 1)
    ymax = int(max(p_y0, p_y1, p_y2) + 1)
    if xmax > 1000: xmax = 1000
    if ymax > 1000: ymax = 1000
    if ((p_x0 - p_x2) * (p_y1 - p_y2) - (p_x1 - p_x2) * (p_y0 - p_y2)) == 0:
        return
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            b0, b1, b2 = bar(x, y, p_x0, p_y0, p_x1, p_y1, p_x2, p_y2)
            if b0 >= 0 and b1 >= 0 and b2 >= 0 and def_cos(x0, y0, z0, x1, y1, z1, x2, y2, z2) < 0:
                z = b0 * z0 + b1 * z1 + b2 * z2
                if z < z_buf[y][x]:
                    img[y][x] = color
                    z_buf[y][x] = z

vertices = []
with open("model2.obj", "r") as file:
    for line in file:
        if line.startswith("v "):
            data = line.split()
            x, y, z = map(float, data[1:])
            x, y, z = turn(0, np.pi, 0, [0, -0.02, 0.1], [x, y, z])
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

polygons = read_model_data('model2.obj')

def def_cos(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    l = [0, 0, 1]
    normal = [(y1 - y2) * (z1 - z0) - (z1 - z2) * (y1 - y0),
              (z1 - z2) * (x1 - x0) - (x1 - x2) * (z1 - z0),
              (x1 - x2) * (y1 - y0) - (y1 - y2) * (x1 - x0)]
    composition = normal[0]*l[0] + normal[1]*l[1] + normal[2]*l[2]
    norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    ll = np.sqrt(l[0]**2 + l[1]**2 + l[2]**2)
    cos = composition/(norm*ll)
    return cos

z_buff = np.full((1000, 1000), np.inf,  dtype=np.float32)
image2 = np.zeros((1000, 1000, 3), dtype=np.uint8)

for i in range(0, len(polygons)):
    p1 = polygons[i][0] - 1
    p2 = polygons[i][1] - 1
    p3 = polygons[i][2] - 1
    x0 = vertices[p1].x 
    y0 = vertices[p1].y 
    z0 = vertices[p1].z 
    x1 = vertices[p2].x 
    y1 = vertices[p2].y 
    z1 = vertices[p2].z
    x2 = vertices[p3].x 
    y2 = vertices[p3].y 
    z2 = vertices[p3].z
    draw(z_buff, image2, x0, y0, z0, x1, y1, z1, x2, y2, z2, [-255 * def_cos(x0, y0, z0, x1, y1, z1, x2, y2, z2), 0, 0])
img = Image.fromarray(image2, mode='RGB')
img = ImageOps.flip(img)
img.save('turn2.png')
img.show()
