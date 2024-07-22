import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from .box import point, Box, space
import copy
import random
from timeit import default_timer as timer
# from .model import d2

## THIS IS THE CODE WITH THE REAL ORACLE FUNCTION (USES THE REAL FORMULA IN MODEL.PY)

def walkBox(p,d2_function, f = -1):
    global Ocount   #Ocount is used to count the number of thime the oracle is called
    corners = []
    n = p.dim-1
    n = 1
    Er = 0.00001
    for y in range(0,n):
        # print("Moving in the {} dim!".format(y+1))
        delta = 1
        pbO = copy.deepcopy(p)
        while d2_function(p) <= 0:         #if the point is outside the box
            Ocount += 1
            print("bad point")
        Ocount += 1
        pb = copy.deepcopy(p)
        insideDis = d2_function(pb)
        Ocount += 1
        while insideDis > 0+Er or insideDis < 0-Er:   #Moving the point to the border of the box
            pb.coord[y] -= insideDis
            insideDis = d2_function(pb)
            Ocount += 1
        po = copy.deepcopy(pb)
        for k in range(y+1,p.dim):  #Starting the loops
            if k != f and f != -1:
                k = f
            # print("Loop in the {} dim:".format(k+1))
            pb = copy.deepcopy(po)
            pi = copy.deepcopy(pb)
            pi.coord[y] += delta
            dir = [k,y]
            s = int(1)
            first = True
            count = 0
            while True:
                if first == True:
                    if count == 0:
                        o = len(corners)
                    if count == 3:
                        first = False
                        count = 0
                    else:
                        count +=1
                dist1 = d2_function(pi)
                Ocount += 1
                pb.coord[dir[0]] += s*dist1
                pi.coord[dir[0]] += s*dist1

                # pb.coord[dir[0]] = round(pb.coord[dir[0]],5)
                # pi.coord[dir[0]] = round(pi.coord[dir[0]],5)

                distpb = d2_function(pb)
                distpi = d2_function(pi)
                Ocount += 1
                if distpi <= 0+Er and distpi >= 0-Er and distpb <= 0+Er and distpb >= 0-Er:     #corner point has been reached
                    Ocount += 2
                    pr = copy.deepcopy(pb)
                    pr.pl = copy.deepcopy(dir)
                    pr.pl[0] = int(round((pr.pl[0] + 1)*s,1))
                    dir[0],dir[1] = dir[1],dir[0]
                    temp_s = pi.coord[dir[0]] - pb.coord[dir[0]]
                    pb = copy.deepcopy(pi)
                    pi.coord[dir[1]] -= s*delta
                    s = temp_s
                    pr.pl[1] = int(round(-(pr.pl[1] + 1)*s,1))
                    corners.append(copy.deepcopy(pr))
                    if o != len(corners)-1:             #connecting the new corner to the previous one
                        corners[-1].points[0] = corners[-2]
                        corners[-2].points[1] = corners[-1]


                if distpb > 0+Er or distpb < 0-Er:                         #intersection point has been reached
                    while distpb > 0+Er or distpb < 0-Er:              #moving the point back to the intersection point
                        pb.coord[dir[0]] -= s*distpb
                        pi.coord[dir[0]] -= s*distpb
                        distpb = d2_function(pb)
                        Ocount += 1
                    # pb.coord[dir[0]] = round(pb.coord[dir[0]],5)
                    # pi.coord[dir[0]] = round(pi.coord[dir[0]],5)
                    pr = copy.deepcopy(pb)
                    pr.pl = copy.deepcopy(dir)
                    pr.pl.append(0)
                    pr.pl[0] = int(round(-(pr.pl[0] + 1)*s,1))
                    dir[0],dir[1] = dir[1],dir[0]
                    temp_s = pb.coord[dir[0]] - pi.coord[dir[0]]
                    pi = copy.deepcopy(pb)
                    pi.coord[dir[1]] += s*delta
                    s = temp_s
                    pr.pl[1] = int(round((pr.pl[1] + 1)*s,1))
                    corners.append(copy.deepcopy(pr))
                    if o != len(corners)-1:             #connecting the new corner to the previous one
                        corners[-1].points[0] = corners[-2]
                        corners[-2].points[1] = corners[-1]

                if len(corners) > o+4:                  #Fail safe
                    if corners[-1].samePoint(corners[o]):
                        break

                if distpb >= 0+Er and distpb <= 0-Er:
                    return []
            #connect the first and last points
            corners = corners[:-1]
            corners[-1].points[1] = corners[o]
            corners[o].points[0] = corners[-1]
            if f != -1:
                break
    return corners

def recon(res, d2_function):
    global Faces
    Faces = []
    n = res[0].dim
    fin = []
    for i in range(0,1):
        # print(">>>>>>>>>>>>")
        # print(i)
        s = 1
        boxes = []
        T = []
        ## getting the front and back of th ith dimension
        while s == 1 or s == -1:
            #print("-------------------------------------------")
            x = []      #stores the the ith position of the face
            x2 = []     #stores the ith position of the face of intersection points
            plain = []  #stores the points corresponding to x
            inter = []  #stores the intersection points corresponding to x2

            # going through the points and storing them depending on their position
            for k in res:
                k.a = 0
                if s*(i+1) in k.pl or -s*(i+1) in k.pl:
                    if s*(i+1) in k.pl and 0 not in k.pl:
                        if k.coord[i] not in x:
                            x.append(k.coord[i])
                            plain.append([])
                        plain[x.index(k.coord[i])].append(k)
                    elif 0 in k.pl:
                        if k.coord[i] not in x2:
                            x2.append(k.coord[i])
                            inter.append([])
                        inter[x2.index(k.coord[i])].append(k)
            # separating the faces at each position
            # print(plain)
            for y in x:
                mess = []
                for p in plain[x.index(y)]:
                    NoBox = 0
                    onetry = 0
                    mix = [0]*n         #stores the points that are on the same face as p and stores them depending on with dimension they represent
                    if p.a == 0:        #Not part of any face that has been found already
                        if p.a == 1:
                            continue
                        p.a = 1
                        mess.append([])
                        if abs(p.pl[1]) == i+1:     #Finding p's opposite point
                            dP = int(p.pl[0])
                            Op = p.points[0]
                        else:
                            dP = int(p.pl[1])
                            Op = p.points[1]
                        Op.a = 1
                        for d in range(0,n):    #going through each dimension to find the points that are on the same face as p
                            if d == i:
                                mix[d] = []
                                continue
                            if d == abs(dP)-1:
                                if dP > 0:
                                    mix[d] = [Op,p]
                                else:
                                    mix[d] = [p,Op]
                                continue
                            low = 0
                            lowP = []
                            high = 100
                            highP = []
                            count = 0
                            TestPoints = plain[x.index(y)]
                            newP = copy.deepcopy(p)
                            while len(lowP) == 0 or len(highP) == 0:    #redo if no points where found but now with the points that where added by mini WalkBox
                                for pp in TestPoints:
                                    if (d+1 in pp.pl or -d-1 in pp.pl) and ((pp.coord[abs(dP)-1]< p.coord[abs(dP)-1] and pp.coord[abs(dP)-1] > Op.coord[abs(dP)-1] and dP > 0) or (pp.coord[abs(dP)-1]> p.coord[abs(dP)-1] and pp.coord[abs(dP)-1] < Op.coord[abs(dP)-1] and dP < 0)):
                                        if pp.pl[1] == s*(i+1):
                                            z = 0
                                        else:
                                            z = 1
                                        # Testing the point pp to see if could be part of the face
                                        if pp in lowP or pp in highP or pp.points[z] in lowP or pp.points[z] in highP:
                                            continue
                                        if ((p.coord[int(abs(pp.pl[z])-1)] > pp.coord[int(abs(pp.pl[z])-1)] or p.coord[int(abs(pp.pl[z])-1)] < pp.points[z].coord[int(abs(pp.pl[z])-1)]) and pp.pl[z] > 0) or ((p.coord[int(abs(pp.pl[z])-1)] < pp.coord[int(abs(pp.pl[z])-1)] or p.coord[int(abs(pp.pl[z])-1)] > pp.points[z].coord[int(abs(pp.pl[z])-1)]) and pp.pl[z] < 0):
                                            continue
                                        if low > pp.coord[d] or high < pp.coord[d]:
                                            continue
                                        inBox = 0
                                        if TestPoints == plain[x.index(y)]:
                                            search = plain[x.index(y)] + TestPoints
                                        else:
                                            search = plain[x.index(y)]

                                        # create a the smallest face that contains p, pp and their opposite points to check if there is any point inside of the face
                                        TestB = Box(n)
                                        for Db in range(0,n):
                                            if Db == i:
                                                continue
                                            if Db == abs(dP)-1:
                                                if p.coord[Db] > Op.coord[Db]:
                                                    TestB.Borders[Db] = [Op.coord[Db],p.coord[Db]]
                                                else:
                                                    TestB.Borders[Db] = [p.coord[Db],Op.coord[Db]]
                                            if Db == d:
                                                if pp.points[z].coord[Db] > pp.coord[Db]:
                                                    TestB.Borders[Db] = [pp.coord[Db],pp.points[z].coord[Db]]
                                                else:
                                                    TestB.Borders[Db] = [pp.points[z].coord[Db],pp.coord[Db]]
                                            else:
                                                if p.coord[Db] > pp.coord[Db]:
                                                    TestB.Borders[Db] = [pp.coord[Db],p.coord[Db]]
                                                else:
                                                    TestB.Borders[Db] = [p.coord[Db],pp.coord[Db]]
                                        for intP in search:
                                            if 0 not in intP.pl:
                                                if TisInBox(TestB,intP) == 1:
                                                    inBox = 1
                                                    break
                                        if inBox == 1:
                                            continue
                                        ########
                                        if len(lowP) == 0 and pp.coord[d]<p.coord[d]:
                                            low = pp.coord[d]
                                            lowP = []
                                            lowP.append(pp)
                                        else:
                                            if low < pp.coord[d] and pp.coord[d]<p.coord[d]:
                                                low = pp.coord[d]
                                                lowP = []
                                                lowP.append(pp)
                                            elif low == pp.coord[d]:
                                                lowP.append(pp)
                                        ##
                                        if len(highP) == 0 and pp.coord[d]>p.coord[d]:
                                            high = pp.coord[d]
                                            highP = []
                                            highP.append(pp)
                                        else:
                                            if high > pp.coord[d] and pp.coord[d]>p.coord[d]:
                                                high = pp.coord[d]
                                                highP = []
                                                highP.append(pp)
                                            elif high == pp.coord[d]:
                                                highP.append(pp)
                                        ########
                                        if len(lowP) == 0 and pp.points[z].coord[d]<p.coord[d]:
                                            low = pp.points[z].coord[d]
                                            lowP = []
                                            lowP.append(pp.points[z])
                                        else:
                                            if low < pp.points[z].coord[d] and pp.points[z].coord[d]<p.coord[d]:
                                                low = pp.points[z].coord[d]
                                                lowP = []
                                                lowP.append(pp.points[z])
                                            elif low == pp.points[z].coord[d]:
                                                lowP.append(pp.points[z])
                                        ##
                                        if len(highP) == 0 and pp.points[z].coord[d]>p.coord[d]:
                                            high = pp.points[z].coord[d]
                                            highP = []
                                            highP.append(pp.points[z])
                                        else:
                                            if high > pp.points[z].coord[d] and pp.points[z].coord[d]>p.coord[d]:
                                                high = pp.points[z].coord[d]
                                                highP = []
                                                highP.append(pp.points[z])
                                            elif high == pp.points[z].coord[d]:
                                                highP.append(pp.points[z])
                                        ########

                                if len(lowP) == 0 or len(highP) == 0:
                                    if onetry == 1:
                                        NoBox = 1
                                        break
                                    tempo = []
                                    TestPoints = []
                                    while True:
                                        if dP > 0:
                                            newP.coord[abs(dP)-1] -= 1
                                            if newP.coord[abs(dP)-1] < Op.coord[abs(dP)-1] or (newP.coord[abs(dP)-1] == Op.coord[abs(dP)-1] and 0 not in Op.pl):
                                                onetry = 1
                                                break
                                        else:
                                            newP.coord[abs(dP)-1] += 1
                                            if newP.coord[abs(dP)-1] > Op.coord[abs(dP)-1] or (newP.coord[abs(dP)-1] == Op.coord[abs(dP)-1] and 0 not in Op.pl):
                                                onetry = 1
                                                break
                                        if s > 0:
                                            newP.coord[i] = p.coord[i] - 1
                                        else:
                                            newP.coord[i] = p.coord[i] + 1
                                        ###################### Mini walkBox################
                                        po = copy.deepcopy(newP)
                                        while d2_function(po) != 0:
                                            clark = 0
                                            po.coord[i] = po.coord[i] + s*d2_function(po)
                                        for side in range(-1,2,2):
                                            dT = d
                                            back = 0
                                            pb = copy.deepcopy(po)
                                            pi = copy.deepcopy(pb)
                                            pi.coord[i] -= s*1
                                            while True:
                                                pb.coord[dT] += side*d2_function(pi)
                                                pi.coord[dT] += side*d2_function(pi)
                                                if d2_function(pi) == 0 and d2_function(pb) == 0:
                                                    if back == 1:
                                                        pb.pl = [s*(i+1),side*(d+1)]
                                                        tempo[-1].points[0] = pb
                                                        break
                                                    pb.pl = [s*(i+1),side*(d+1)]
                                                    tempo.append(copy.deepcopy(pb))
                                                    pb = copy.deepcopy(pi)
                                                    pi.coord[dT] -= side*1
                                                    dT = i
                                                    side = -s
                                                    back = 1

                                                if d2_function(pb) != 0:
                                                    while d2_function(pb) != 0:
                                                        pb.coord[dT] -= side*d2_function(pb)
                                                        pi.coord[dT] -= side*d2_function(pb)
                                                    if back == 1:
                                                        pb.pl = [s*(i+1),side*(d+1),0]
                                                        tempo[-1].points[0] = pb
                                                        break
                                                    pb.pl = [s*(i+1),side*(d+1),0]
                                                    tempo.append(copy.deepcopy(pb))
                                                    break
                                        # tempo = walkBox(newP,S,d)
                                        tempo[0].points[1] = tempo[1]
                                        tempo[1].points[1] = tempo[0]
                                        for te in tempo:
                                            if te.coord[i] == y:
                                                TestPoints.append(te)
                                        tempo = []
                            if len(lowP) == 0 or len(highP) == 0:
                                NoBox = 1
                                break
                            mix[d] = lowP + highP
                        if NoBox == 1:
                            mess = mess[:-1]
                            continue
                        for m in mix:
                            for m1 in m:
                                m1.a = 1
                        for pp in plain[x.index(y)]:
                                if pp in mess[-1] or pp.points[0] in mess[-1] or pp.points[1] in mess[-1]:
                                    continue
                                if (dP in pp.pl or -dP in pp.pl):
                                    ppB = pp
                                    for count in range (0,2):
                                        result = True
                                        if abs(pp.pl[0]) == i+1:
                                            ppBOp = 1
                                        else:
                                            ppBOp = 0
                                        if count == 1:
                                            ppB = pp.points[ppBOp]
                                        if (ppB.coord[abs(dP)-1] == mix[abs(dP)-1][0].coord[abs(dP)-1] and (-abs(dP) in ppB.pl or (abs(dP) in ppB.pl and 0 in ppB.pl))) or (ppB.coord[abs(dP)-1] == mix[abs(dP)-1][1].coord[abs(dP)-1] and (abs(dP) in ppB.pl or (-abs(dP) in ppB.pl and 0 in ppB.pl))):
                                            for bP in range(0,n):
                                                if bP == i or bP == abs(dP)-1:
                                                    continue
                                                if len(mix[bP]) > 1:
                                                    if ppB.coord[bP] < mix[bP][0].coord[bP] or ppB.coord[bP] > mix[bP][-1].coord[bP]:
                                                        result = False
                                                        break
                                                else:
                                                    continue
                                            if result == True:
                                                ppB.a = 1
                                                mess[-1].append(ppB)
                                                if count == 0:
                                                    plain[x.index(y)][plain[x.index(y)].index(pp)].a = 1
                                                else:
                                                    if pp.points[ppBOp] in plain[x.index(y)]:
                                                        plain[x.index(y)][plain[x.index(y)].index(pp.points[ppBOp])].a = 1
                        for cou in mix:
                            if p not in cou:
                                mess[-1] += cou
                plain[x.index(y)] = mess
                for MESS in mess:
                    Faces += MESS
            #going through each face to find the borders
            # print("part3")
            # print(x)
            # print("waaw")
            for y in x:
                Bo = Box(n)          #intinalize the box and set half the ith borders
                if s < 0:
                    Bo.Borders[i] = [y,100]
                    Bo.change[i][0] = 1
                else:
                    Bo.Borders[i] = [0,y]
                    Bo.change[i][1] = 1
                temp = 10000000000000          #temp represents the potential second ith border in case we faild to find the other side because of an inter section(the other face is inside another box)
                tempC = 0
                for lP in range(0,len(plain[x.index(y)])):

                    B = copy.deepcopy(Bo)
                    #going through the points in the face to get the borders
                    for p in plain[x.index(y)][lP]:
                        #find the dimension of the 2 opposite points. One of the points is on the same face and the other is on the opposit face.
                        # The point on the same face will be used incase a side is missing from the face (one of the faces is determind by intersection points)
                        # The point on the opposite face will be used incase the opposite face is missing (inside of another box)
                        if 0 not in p.points and 0 not in p.pl:
                            if abs(p.pl[1]) == i+1:
                                if 0 not in p.points[1].pl and tempC != 1:
                                    temp = p.points[1].coord[int(abs(p.pl[1])-1)]
                                    tempC = 1
                                elif abs(p.points[1].coord[int(abs(p.pl[1])-1)]-p.coord[int(abs(p.pl[1])-1)])<abs(temp-p.coord[abs(int(abs(p.pl[1])-1))]):
                                    if 0 not in p.points[1].pl:
                                        temp = p.points[0].coord[int(abs(p.pl[0])-1)]
                                    elif 0 in p.points[1].pl and tempC != 1:
                                        temp = p.points[1].coord[int(abs(p.pl[1])-1)]
                                        tempC = 0.5
                            else:
                                if 0 not in p.points[0].pl and tempC != 1:
                                    temp = p.points[0].coord[int(abs(p.pl[0])-1)]
                                    tempC = 1
                                elif abs(p.points[0].coord[int(abs(p.pl[0])-1)]-p.coord[int(abs(p.pl[0])-1)])<abs(temp-p.coord[int(abs(p.pl[0])-1)]):
                                    if 0 not in p.points[0].pl:
                                        temp = p.points[1].coord[int(abs(p.pl[1])-1)]
                                    elif 0 in p.points[0].pl and tempC != 1:
                                        temp = p.points[0].coord[int(abs(p.pl[0])-1)]
                                        tempC = 0.5
                        # find the dimension this points limits (ex: if p.pl = [-2,1] and i = 1 then p is in face in the 1 dimension and is the left limit in the 2 dimension)
                        if p.pl[1] == s*(i+1):
                            z = int(abs(p.pl[0])-1)
                        else:
                            z = int(abs(p.pl[1])-1)

                        # check if its left limit or a right limit
                        if z+1 in p.pl or -(z+1) in p.pl:
                            if max(B.Borders[z][1],p.coord[z]) != B.Borders[z][1] or B.change[z][1] == -1:
                                B.Borders[z][1] = p.coord[z]    #set the right z Border of the box
                                if 0 in p.pl:
                                    B.change[z][1] = 0.5
                                    B.Borders[z][1] += 1
                                else:
                                    B.change[z][1] = 1

                        #same thing but if its a left limit
                            if min(B.Borders[z][0],p.coord[z]) != B.Borders[z][0] or B.change[z][0] == -1:
                                B.Borders[z][0] = p.coord[z]
                                if 0 in p.pl:
                                    B.change[z][0] = 0.5
                                    B.Borders[z][0] -= 1
                                else:
                                    B.change[z][0] = 1
                    if tempC == 1:
                        if s < 0:
                            B.Borders[i][1] = temp
                            B.change[i][1] = tempC
                        else:
                            B.Borders[i][0] = temp
                            B.change[i][0] = tempC
                    ###################################################################################################################################
                    # This is to make sure no intersection point are in the boxes
                    test = 0
                    for y1 in x2:
                        if (y > y1 and B.Borders[i][0] < y1 and s > 0) or (y < y1 and B.Borders[i][1] > y1 and s < 0):
                            for pI in inter[x2.index(y1)]:
                                    if B.isInBox(pI) == 1:
                                        test = 1
                                        if s < 0:
                                            B.Borders[i][1] = pI.coord[i]
                                            B.change[i][1] = 1
                                        else:
                                            B.Borders[i][0] = pI.coord[i]
                                            B.change[i][0] = 1
                    ###################################################################################################################################
                    if tempC == 0.5 and test == 0:
                        if s < 0:
                            B.Borders[i][1] = temp+1
                            B.change[i][1] = 1
                        else:
                            B.Borders[i][0] = temp-1
                            B.change[i][0] = 1
                    bad = 0
                    j= 0
                    while j < len(fin):           # if the new box matches with any of the end result boxes that means its either a useless repeat or it could be a more accurate version of it.
                        if fin[j].redundant(B) == True:
                            fin.remove(fin[j])      # we remove the matching boxes to get less garbage answers
                            continue
                        elif B.redundant(fin[j]) == True:    # this means the new box is inside an existing box so its just useless
                            bad = 1
                            break
                        j += 1
                    for testp in res:
                        if B.isInBox(testp) == 1:
                            bad = 1
                            break
                    if bad == 0:
                        fin.append(copy.deepcopy(B))   #If it doesn't match any of the end result boxes we can add it to the list
            s -= 2  # we went through all the s = 1 faces now we move to s = -1
    return fin

def sample(U,N):
    # n = U.dim
    # result = []
    # Area = []
    # total = 0
    # for boxes in U.Boxes:
    #     a = 1
    #     for i in range(0,n):
    #         a *= boxes.Borders[i][1] - boxes.Borders[i][0]
    #     Area.append(a)
    #     total += a

    # for boxes in U.Boxes:
    #     nbPoints = int(N*Area[U.Boxes.index(boxes)]/total + 1)
    #     for i in range(0,nbPoints):
    #         bre = 0
    #         p = point(n)
    #         for j in range(0,n):
    #             if float(boxes.Borders[j][0]+1) >= float(boxes.Borders[j][1])-1:
    #                 p.coord[j] = boxes.Borders[j][0] + 0.5
    #             else:
    #                 p.coord[j] = random.uniform(boxes.Borders[j][0]+1,boxes.Borders[j][1]-1)
    #         if bre == 0:
    #             result.append(p)
    n = U.dim
    result = []
    Area = []
    total = 0
    for boxes in U.Boxes:
        a = 1
        for i in range(0,n):
            a *= boxes.Borders[i][1] - boxes.Borders[i][0]
        Area.append(a)
        total += a

    for boxes in U.Boxes:
        nbPoints = int(N*Area[U.Boxes.index(boxes)]/total + 1)
        for i in range(0,nbPoints):
            bre = 0
            p = point(n)
            for j in range(0,n):
                if int(boxes.Borders[j][0]+1) >= int(boxes.Borders[j][1])-1:
                    p.coord[j] = boxes.Borders[j][0] + 0.5
                else:
                    p.coord[j] = random.randint(boxes.Borders[j][0]+1,boxes.Borders[j][1]-1)
            if bre == 0:
                result.append(p)
    return result

def buildset(n,N,x,y):
    U = space(n)
    for i in range(0,N):
        No = 0
        B = Box(n)
        for j in range(0,n):
            B0 = random.uniform(x,y-5)
            B1 = random.uniform(B0+5,y)
            B.Borders[j] = [B0,B1]
        for i in U.Boxes:
            if i.redundant(B) == True:
                No = 1
                break
            if B.redundant(i) == True:
                U.Boxes.remove(i)
        if No == 0:
            U.addBoxes(B)
    return U

def TisInBox(B,x):
    if B.dim != x.dim:
        return False
    else:
        ans = 1
        for i in range(0,B.dim):
            if x.coord[i] < B.Borders[i][0] or  x.coord[i] > B.Borders[i][1]:
                return 0
            elif x.coord[i] == B.Borders[i][0] or  x.coord[i] == B.Borders[i][1]:
                ans = 0.5
    return ans

def sampleC(U,N):
    n = U.dim
    result = []
    Area = []
    total = 0
    for boxes in U.Boxes:
        a = 1
        for i in range(0,n):
            a *= boxes.Borders[i][1] - boxes.Borders[i][0]
        Area.append(a)
        total += a

    for boxes in U.Boxes:
        nbPoints = int(N*Area[U.Boxes.index(boxes)]/total + 1)
        if nbPoints > N/3:
            nbPoints = int(N/3)
        for i in range(0,nbPoints):
            bre = 0
            p = point(n)
            for j in range(0,n):
                if float(boxes.Borders[j][0]+1) >= float(boxes.Borders[j][1])-1:
                    p.coord[j] = boxes.Borders[j][0] + 0.5
                else:
                    p.coord[j] = random.uniform(boxes.Borders[j][0]+1,boxes.Borders[j][1]-1)
            # p.b = 1
            # for i in result:
            #     if i.samePoint(p) == 1:
            #         bre = 1
            #         # print("same point")
            #         break
            if bre == 0:
                result.append(p)
    return result

Ocount = 0
def FindBox(n,Box, d1_function, d2_function):
    n = n
    over = 0
    under = 0
    overA = 0
    underA = 0
    total = 0
    avrgWalk = 0
    avrgRecon = 0
    


    B = Box

    ALL = space(n)
    ALL.addBoxes(B)

    # B1 = Box(10)
    # B2 = Box(10)
    # B3 = Box(10)
    # B4 = Box(10)
    # B5 = Box(10)
    # B6 = Box(10)
    # B7 = Box(10)
    # B8 = Box(10)
    # B9 = Box(10)
    # B10 = Box(10)
    # B11 = Box(10)
    # B12 = Box(10)
    # B13 = Box(10)
    # B14 = Box(10)
    # B15 = Box(10)
    # B16 = Box(10)
    # B17 = Box(10)

    # B1.Borders = [[-20,20],[-20,20],[-20,20],[-20,20],[-20,3],[10,20],[10,20],[-20,20],[-20,20],[-20,20]]
    # B2.Borders = [[-20,20],[-20,20],[-20,20],[-20,20],[-20,3],[-20,3],[10,20],[-20,20],[-20,20],[-20,20]]
    # B3.Borders = [[-20,20],[-20,20],[-20,20],[-20,20],[10,20],[-20,3],[-20,3],[-20,20],[-20,20],[-20,20]]
    # B4.Borders = [[-20,20],[-20,20],[-20,20],[-20,20],[-20,3],[-20,3],[-20,3],[-20,20],[-20,20],[-20,20]]
    # B5.Borders = [[-20,20],[-20,20],[0,10],[-20,20],[-20,3],[10,20],[-20,20],[-20,20],[-20,20],[-20,20]]
    # B6.Borders = [[-20,20],[-20,20],[-20,20],[-20,20],[10,20],[10,20],[-20,3],[-20,20],[-20,20],[-20,20]]
    # B7.Borders = [[-20,20],[0,10],[-20,20],[-20,20],[-20,20],[10,20],[-20,3],[-20,20],[-20,20],[-20,20]]
    # B8.Borders = [[-20,20],[-20,20],[0,10],[-20,20],[-20,3],[-20,3],[-20,20],[-20,20],[-20,20],[-20,20]]
    # B9.Borders = [[-20,20],[-20,20],[-20,20],[-20,20],[-20,3],[10,20],[-20,3],[-20,20],[-20,20],[-20,20]]
    # B10.Borders = [[-20,20],[-20,20],[-20,20],[-20,20],[10,20],[-20,3],[10,20],[-20,20],[-20,20],[-20,20]]
    # B11.Borders = [[-20,20],[-20,20],[-20,20],[-20,20],[10,20],[10,20],[10,20],[-20,20],[-20,20],[-20,20]]
    # B12.Borders = [[-20,20],[0,10],[-20,20],[-20,20],[-20,20],[-20,3],[10,20],[-20,20],[-20,20],[-20,20]]
    # B13.Borders = [[-20,20],[-20,20],[0,10],[-20,20],[10,20],[-20,3],[-20,20],[-20,20],[-20,20],[-20,20]]
    # B14.Borders = [[-20,20],[0,10],[-20,20],[-20,20],[-20,20],[-20,3],[-20,3],[-20,20],[-20,20],[-20,20]]
    # B15.Borders = [[-20,20],[-20,20],[0,10],[-20,20],[10,20],[10,20],[-20,20],[-20,20],[-20,20],[-20,20]]
    # B16.Borders = [[-20,20],[0,10],[-20,20],[-20,20],[-20,20],[10,20],[10,20],[-20,20],[-20,20],[-20,20]]
    # B17.Borders = [[-20,20],[0,10],[0,10],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20],[-20,20]]

    # Ex = space(10)
    # Ex.Boxes = [B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16,B17]

    start = timer()
    Sam =  sample(ALL, 50000)
    end = timer()
    print("sample Time:", end - start)

    result = []
    start = timer()
    print("|------------------------------WalkBox---------------------------------------------|")
    for i in Sam:
        if d2_function(i) > 0:
            print(Sam.index(i))
            result += walkBox(i, d2_function=d2_function)
    end = timer()
    print("|----------------------------------------------------------------------------------|")
    print("walkBox Time:", end - start)
    avrgWalk += end - start
    print("|----------------------------------------------------------------------------------|")
    print("recon:")
    rec = space(n)
    start = timer()
    rec.Boxes = recon(result, d2_function=d2_function)
    end = timer()
    print("|----------------------------------------------------------------------------------|")
    print("recon Time:", end - start)
    avrgWalk += end - start
    print("|----------------------------------------------------------------------------------|")


    Sam1 = sample(rec, 1000)

    badp = 0
    for i in Sam1:
        if d2_function(i) < 0:
            print("BAD POINT")
            print(i.coord)
            print(d2_function(i))
            badp +=1
    badp2 = 0

    Sam = sample(ALL, 1000)
    for i in Sam:
        if d2_function(i) > 0 and d1_function(i,rec) < 0:
            print("BAD POINT")
            print(i.coord)
            badp2 +=1

    print("OVER BAD POINTS: " + str(badp))
    print("UNDER BAD POINTS: " + str(badp2))
    print("FINAL:")
    for i in rec.Boxes:
        print(i)
        pass

    return rec

# B = Box(3)
# B.Borders = [[-20,20],[-20,20],[-20,20]]
# result = FindBox(3,B, d1, d2)



# p = point(3)
# p.coord = [1,1,1]
# print(d1(p,result))