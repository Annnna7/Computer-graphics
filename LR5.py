class Quaternion:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
    def __add__(self, other):
        return Quaternion(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)
    
    def __mul__(self, other):
        return Quaternion(self.a * other.a - self.b * other.b - self.c * other.c - self.d * other.d,
                          self.a * other.b + self.b * other.a + self.c * other.d - self.d * other.c,
                          self.a * other.c - self.b * other.d + self.c * other.a + self.d * other.b,
                          self.a * other.d + self.b * other.c - self.c * other.b + self.d * other.a)
    
    def conj(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)
    
    def __str__(self):
        return f"{self.a} + {self.b}i + {self.c}j + {self.d}k"
    
a = Quaternion(0, 1, 0, 0)
b = Quaternion(0, 0, 1, 0)
c = a * b
print(c)