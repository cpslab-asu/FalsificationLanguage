import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from itertools import product, combinations
from timeit import default_timer as timer

global delta

# THIS FILE IS USED TO CREATE THE BOXES AND THE POINTS CLASSES
# ALSO THE "FACK" ORACLE FUNCTION

delta = 0.001
class point:
    def __init__(self,n):
        self.dim = n
        self.coord = [0]*n
        self.pl = [0,0]
        self.points = [self,self]
        self.faces = 0
        self.a = 0
        self.b = 0
        self.r = 0

    def defineCoords(self):
        for  i in range(0,self.dim):
            b = input("Dimention {}: ".format(i+1))
            self.coord.append(int(b))

    def samePoint(self,x):
        for i in range(0,self.dim):
            if self.coord[i] < x.coord[i] - 0.00001 or self.coord[i] > x.coord[i] + 0.00001:   
                return False
        return True

    def plot(self,ax,a):
        if self.a == 1:
            c = "yellow"
            s = 10
        else:
            c = "red"
            s = 5
        ax.plot(self.coord[0], self.coord[1], marker="o", markersize=s, markeredgecolor=c, markerfacecolor=c)

    def plot3D(self,ax,a):
        if a == 1:
            c = "green"
            s = 100
        else:
            if self.b == 1:
                c = "yellow"
                s = 100
            else:
                c = "red"
                s = 100
                if(0 in self.pl):
                    c = "black"
        ax.scatter(self.coord[0], self.coord[1], self.coord[2], c=c, marker='o', s=s)

class Box:
    def __init__(self,n):
        self.dim = n
        self.Borders = []
        self.change = []
        for i in range(0,n):
            self.Borders.append([-1,101])
            self.change.append([-1,-1])

    def defineBorders(self):
        for  i in range(0,self.dim):
            print("In dimention {}:".format(i+1))
            b1 = input("x > ")
            b2 = input("x < ")
            self.Borders.append((int(b1),int(b2)))
    
    def repeatBorders(self,x,y):
        for i in range(0,self.dim):
            self.Borders.append([x,y])
            self.change.append([-1,-1])

    def match(self,x):
        for i in range(0,self.dim):
            if (self.Borders[i][0] < x.Borders[i][0] and ((self.change[i][0] != 0 and self.change[i][0] != 0.5) and (x.change[i][0] != 0 and x.change[i][0] != 0.5))) or (self.Borders[i][1] > x.Borders[i][1] and ((self.change[i][1] != 0 and self.change[i][1] != 0.5) and (x.change[i][1] != 0 and x.change[i][1] != 0.5))):
                return False
            if (self.change[i][0] != 0 and x.change[i][1] != 0):
                if self.Borders[i][0] >= x.Borders[i][1]:
                    return False
            if (self.change[i][1] != 0 and x.change[i][0] != 0):
                if self.Borders[i][1] <= x.Borders[i][0]:
                    return False
        return True
    
    def redundant(self,x):
        for i in range(0,self.dim):
            if (self.Borders[i][0] < x.Borders[i][0]) or (self.Borders[i][1] > x.Borders[i][1]):
                return False
        return True
    
    def merge(self,x):
        for i in range(0,self.dim):
            if self.change[i][0] == 0:
                self.Borders[i][0] = x.Borders[i][0]
                self.change[i][0] = x.change[i][0]
            if self.change[i][1] == 0:
                self.Borders[i][1] = x.Borders[i][1]
                self.change[i][1] = x.change[i][1]
            if self.change[i][0] == 0.5 and self.Borders[i][0] > x.Borders[i][0]:
                self.Borders[i][0] = x.Borders[i][0]
                self.change[i][0] = x.change[i][0]
            if self.change[i][1] == 0.5 and self.Borders[i][1] < x.Borders[i][1]:
                self.Borders[i][1] = x.Borders[i][1]
                self.change[i][1] = x.change[i][1]

    def plot(self,ax):
        ax.add_patch(Rectangle((self.Borders[0][0], self.Borders[1][0]),self.Borders[0][1]-self.Borders[0][0], self.Borders[1][1] - self.Borders[1][0],
                                edgecolor = 'blue',
                                fill=False,
                                lw=5))

    def plot3D(self,ax,b):
        x, y, z = np.indices((40, 40, 40))
        axes = [40, 40, 40]
        cube1 = (x >= self.Borders[0][0]+20) & (y >= self.Borders[1][0]+20) & (z >= self.Borders[2][0]+20) & (x < self.Borders[0][1]+20) & (y < self.Borders[1][1]+20) & (z < self.Borders[2][1]+20)
        colors = np.empty(axes + [4], dtype=np.float32)
        if b == 0:
            colors[cube1] = [0, 0, 1, 0.7]
        else:
            colors[cube1] = [1, 0, 0, 0.7]
        ax.voxels(cube1, facecolors=colors, edgecolor= 'grey')

    def isInBox(self,x):
        if self.dim != x.dim:
            return False
        else:
            ans = 1
            for i in range(0,self.dim):
                if x.coord[i] < self.Borders[i][0]-delta or  x.coord[i] > self.Borders[i][1]+ delta:
                    return 0
                elif (x.coord[i] >= self.Borders[i][0]-delta and x.coord[i] <= self.Borders[i][0]+delta)  or  (x.coord[i] >= self.Borders[i][1]-delta and x.coord[i] <= self.Borders[i][1]+delta):
                    ans = 0.5
        return ans

    def minDist(self,x):
        dist = 10000000
        if self.dim != x.dim:
            return "doesn't work!"
        else:
            s = -1
            if self.isInBox(x) == 1:
                s = 1
            for i in range(0,self.dim):
                    dist = min(abs(x.coord[i] - self.Borders[i][0]),  abs(self.Borders[i][1] - x.coord[i]), dist)
            return s*dist

    def __str__(self):
        result = ""
        for i in self.Borders:
            result += "["+str(i[0])+","+str(i[1])+"],"
        return "["+result[:-1]+"]"

class space:
    def __init__(self,n):
        self.dim = n
        self.Boxes = []

    def addBoxes(self, box):
        self.Boxes.append(box)

    def match(self,x):
        for i in self.Boxes:
            ma = False
            for y in x.Boxes:
                if i.match(y):
                    ma = True
            if ma == False:
                return False
        return True

    def plot(self,ax):
        plt.ylim(0,20)
        plt.xlim(0,20)
        plt.grid()
        for i in self.Boxes:
            i.plot(ax)
        plt.show()

    def plot3D(self,ax,x,b):
        plt.ylim(0,40)
        plt.xlim(0,40)
        ax.set_zlim(0,40)
        plt.grid()
        if x == 1:
            for i in self.Boxes:
                i.plot3D(ax,b)
        plt.show()


def d1(x,U):
    dist = -100000
    for i in U.Boxes:
        dist = max(dist,i.minDist(x))
    return dist