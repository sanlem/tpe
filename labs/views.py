from django.shortcuts import render
import numpy as np
import random


class TableRow:
    def __init__(self, A, rand_range):
        # first generate 3 random X
        self.A = [float(a) for a in A]
        self.X = [float(random.randrange(-rand_range, rand_range)) for i in range(3)]
        self.Y = self._calculate_y()
        self.preferred = False
        print(self.A)

    def _calculate_y(self):
        return self.A[0] + self.A[1] * self.X[0] + self.A[2] * self.X[1] + self.A[3] * self.X[2]


class Table:
    def __init__(self, row_number, A, rand_range):
        self.rows = []
        self.rand_range = rand_range
        for i in range(row_number):
            self.rows.append(TableRow(A, rand_range))
        self.A = [float(a) for a in A]

        self.X0 = []
        self.dx = []
        self.Xn = []
        x0 = [row.X[0] for row in self.rows]
        self.X0.append((max(x0) + min(x0)) / 2)
        self.dx.append(self.X0[-1] - min(x0))

        x1 = [row.X[1] for row in self.rows]
        self.X0.append((max(x1) + min(x1)) / 2)
        self.dx.append(self.X0[-1] - min(x1))

        x2 = [row.X[2] for row in self.rows]
        self.X0.append((max(x2) + min(x2)) / 2)
        self.dx.append(self.X0[-1] - min(x2))

        self.Yet = self.A[0] + self.A[1] * self.X0[0] + self.A[2] * self.X0[1] + self.A[3] * self.X0[2]

        for row in self.rows:
            xn = [(row.X[i] - self.X0[i]) / self.dx[i] for i in range(3)]
            row.xn = xn
            # self.Xn.append(xn)

        maximum = (self.rows[0].Y - self.Yet) ** 2
        pos_max = 0
        pos = 0
        for row in self.rows:
            row.value = (row.Y - self.Yet) ** 2
            if maximum < row.value:
                maximum = row.value
                pos_max = pos
            pos += 1

        self.rows[pos_max].preferred = True


def lab1(request):
    if request.method == "GET":
        return render(request, "lab1.html", {})
    else:
        a = request.POST.get("a_coefs").split(" ")
        rand_range = request.POST.get("rand_range")
        table = Table(8, a, int(rand_range))
        return render(request, "lab1.html", {"table": table})


class NormRow:
    def __init__(self, num, minY, maxY, m, minX=None, maxX=None):
        if num == 0:
            self.X = [-1, -1]
            # self.X = [minX] * 2
        elif num == 1:
            self.X = [1, -1]
            # self.X = [maxX, minX]
        elif num == 2:
            self.X = [-1, 1]
            # self.X = [minX, maxX]

        self.Y = [random.randint(minY, maxY) for _ in range(m)]

        # calc average y
        self.Yavg = sum(self.Y) / m
        self.dispersion = sum([(y - self.Yavg) ** 2 for y in self.Y]) / m


class Determinant:

    def __call__(self, x):
        koefs = [x[0][0] * x[1][1] * x[2][2],
                 x[1][0] * x[2][1] * x[0][2],
                 x[0][1] * x[1][2] * x[2][0],
                 -(x[2][0] * x[1][1] * x[0][2]),
                 -(x[2][1] * x[1][2] * x[0][0]),
                 -(x[1][0] * x[0][1] * x[2][2])]
        return sum(koefs)


def lab2(request):
    X1min = -25
    X1max = -5
    X2min = -15
    X2max = 35
    Ymax = (30 - 26) * 10
    Ymin = (20 - 26) * 10
    # X1min = -25
    # X1max = 75
    # X2min = 5
    # X2max = 40
    # Ymax = 20
    # Ymin = 9

    m = 10000
    norm_table = [NormRow(i, Ymin, Ymax, m) for i in range(3)]

    main_vidh = ((4 * m - 4) / (m * m - 4)) ** 0.5

    Fuv = [norm_table[0].dispersion / norm_table[1].dispersion,
           norm_table[2].dispersion / norm_table[0].dispersion,
           norm_table[2].dispersion / norm_table[1].dispersion]

    m_koef = m - 2 / m
    Sigmauv = [fuv * m_koef for fuv in Fuv]
    Ruv = [(s - 1) * main_vidh for s in Sigmauv]

    mx1 = sum([row.X[0] for row in norm_table]) / len(norm_table)
    mx2 = sum([row.X[1] for row in norm_table]) / len(norm_table)
    mY = sum([row.Yavg for row in norm_table]) / len(norm_table)

    a1 = sum([row.X[0] ** 2 for row in norm_table]) / len(norm_table)
    a2 = []
    for i in range(2):
        a2.append(norm_table[0].X[i] * norm_table[1].X[i])

    a2 = sum(a2) / len(a2)
    a3 = sum([row.X[1] for row in norm_table]) / len(norm_table)

    a11 = sum([row.X[0] * row.Yavg for row in norm_table]) / len(norm_table)
    a22 = sum([row.X[1] * row.Yavg for row in norm_table]) / len(norm_table)

    det = Determinant()
    denominator = det([[1, mx1, mx2], [mx1, a1, a2], [mx2, a2, a3]])
    b0 = det([[mY, mx1, mx2], [a11, a1, a2], [a22, a2, a3]]) / denominator
    b1 = det([[1, mY, mx2], [mx1, a11, a2], [mx2, a22, a3]]) / denominator
    b2 = det([[1, mx1, mY], [mx1, a1, a11], [mx2, a2, a22]]) / denominator

    check = [b0 + b1 * row.X[0] + b2 * row.X[1] for row in norm_table]

    dx1 = abs(X1max - X1min) / 2
    dx2 = abs(X2max - X2min) / 2
    X10 = (X1max - X1min) / 2
    X20 = (X2max - X2min) / 2

    A0 = b0 - b1 * (X10 / dx1) - b2 * (X20 / dx2)
    A1 = b1 / dx1
    A2 = b2 / dx2

    final_check = []
    final_check.append([A0 + A1 * X1min + A2 * X2min])
    final_check.append([A0 + A1 * X1max + A2 * X2min])
    final_check.append([A0 + A1 * X1min + A2 * X2max])
    # final_check.append([A0 + A1 * (-1) + A2 * (-1)])
    # final_check.append([A0 + A1 * X1max + A2 * X2min])
    # final_check.append([A0 + A1 * X1min + A2 * X2max])

    return render(request, "lab2.html", {
        'X1min': X1min,
        'X1max': X1max,
        'X2min': X2min,
        'X2max': X2max,
        'Ymin': Ymin,
        'Ymax': Ymax,
        'norm_table': norm_table,
        'main_vidh': main_vidh,
        'Fuv': Fuv,
        'Sigmauv': Sigmauv,
        'Ruv': Ruv,
        'mx1': mx1,
        'mx2': mx2,
        'mY': mY,
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'a11': a11,
        'a22': a22,
        'b0': b0,
        'b1': b1,
        'b2': b2,
        'dx1': dx1,
        'dx2': dx2,
        'x10': X10,
        'x20': X20,
        'A0': A0,
        'A1': A1,
        'A2': A2,
        'final_check': final_check,
        'check': check
    })


def lab3(request):
    X1min = -25
    X1max = -5
    X2min = -15
    X2max = 35
    X3min = -5
    X3max = 60

    Xmaxavg = (X1max + X2max + X3max) / 3
    Xminavg = (X1min + X2min + X3min) / 3
    Ymin = 200 + Xminavg
    Ymax = 200 + Xmaxavg

    Y1 = [float(random.randrange(Ymin, Ymax)) for _ in range(3)]
    Y2 = [float(random.randrange(Ymin, Ymax)) for _ in range(3)]
    Y3 = [float(random.randrange(Ymin, Ymax)) for _ in range(3)]
    Y4 = [float(random.randrange(Ymin, Ymax)) for _ in range(3)]
    mx1 = (X1min * 2 + X1max * 2) / 4
    mx2 = (X2min * 2 + X2max * 2) / 4
    mx3 = (X3min * 2 + X3max * 2) / 4

    Yavg = [sum(Y1) / 3, sum(Y2) / 3, sum(Y3) / 3, sum(Y4) / 3]
    mY1 = sum(Y1) / 3
    mY2 = sum(Y2) / 3
    mY3 = sum(Y3) / 3
    mY4 = sum(Y4) / 3
    mY = sum(Yavg) / 4

    a1 = (X1min * Yavg[0] + X1min * Yavg[1] + X1max * Yavg[2] + X1max * Yavg[3]) / 4
    a2 = (X2min * Yavg[0] + X2max * Yavg[1] + X2min * Yavg[2] + X2max * Yavg[3]) / 4
    a3 = (X3min * Yavg[0] + X3max * Yavg[1] + X3max * Yavg[2] + X3min * Yavg[3]) / 4

    a11 = ((X1min ** 2) * 2 + (X1max ** 2) * 2) / 4
    a22 = ((X2min ** 2) * 2 + (X2max ** 2) * 2) / 4
    a33 = ((X3min ** 2) * 2 + (X3max ** 2) * 2) / 4

    a12 = (X1min * X2min + X1min * X2max + X1max * X2min + X1max * X2max) / 4
    a21 = a12
    a13 = (X3min ** 2 + X3max ** 2 + X3max ** 2 + X3min ** 2) / 4
    a31 = a13
    a23 = a13
    a32 = a13
    b = []
    a = np.array([[1, 2], [3, 4]])
    np.linalg.det(a)
    num = np.array([[mY, mx1, mx2, mx3], [a1, a11, a12, a13], [a2, a12, a22, a32], [a3, a13, a23, a33]])
    denom = np.matrix([[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a32], [mx3, a13, a23, a33]])
    n = np.linalg.det(num)
    d = np.linalg.det(denom)
    b0 = np.linalg.det(num) / np.linalg.det(denom)
    b.append(b0)

    num = np.array([[1, mY, mx2, mx3], [mx1, a1, a12, a13], [mx2, a2, a22, a32], [mx3, a3, a23, a33]])
    b1 = np.linalg.det(num) / np.linalg.det(denom)
    b.append(b1)

    num = np.array([[1, mx1, mY, mx3], [mx1, a11, a1, a13], [mx2, a12, a2, a32], [mx3, a13, a23, a33]])
    b2 = np.linalg.det(num) / np.linalg.det(denom)
    b.append(b2)

    num = np.array([[1, mx1, mx2, mY], [mx1, a11, a12, a1], [mx2, a12, a22, a2], [mx3, a13, a23, a3]])
    b3 = np.linalg.det(num) / np.linalg.det(denom)
    b.append(b3)

    test1 = b0 + b1 * X1min + b2 * X2min + b3 * X3min
    test2 = b0 + b1 * X1min + b2 * X2max + b3 * X3max
    test3 = b0 + b1 * X1max + b2 * X2min + b3 * X3max
    test4 = b0 + b1 * X1max + b2 * X2max + b3 * X3min
    # matrix for statistic
    X0 = [1] * 4
    X1 = [-1] * 2 + [1] * 2
    X2 = [-1, 1, -1, 1]
    X3 = [-1, 1, 1, -1]
    y1 = [float(random.randrange(Ymin, Ymax)) for _ in range(3)]
    y2 = [float(random.randrange(Ymin, Ymax)) for _ in range(3)]
    y3 = [float(random.randrange(Ymin, Ymax)) for _ in range(3)]
    y4 = [float(random.randrange(Ymin, Ymax)) for _ in range(3)]
    yavg = [sum(y1) / 3, sum(y2) / 3, sum(y3) / 3, sum(y4) / 3]

    my1 = sum(y1) / 3
    my2 = sum(y2) / 3
    my3 = sum(y3) / 3
    my4 = sum(y4) / 3

    dy1 = sum([(y - my1) ** 2 for y in y1]) / 3
    dy2 = sum([(y - my2) ** 2 for y in y2]) / 3
    dy3 = sum([(y - my3) ** 2 for y in y3]) / 3
    dy4 = sum([(y - my4) ** 2 for y in y4]) / 3

    Gp = max(dy1, dy2, dy3, dy4) / sum([dy1, dy2, dy3, dy4])

    Sb = sum([dy1, dy2, dy3, dy4]) / 4
    S2bs = Sb / 12
    Sbs = S2bs ** 0.5

    B0 = sum([yavg[i] * X0[i] for i in range(4)]) / 4
    B1 = sum([yavg[i] * X1[i] for i in range(4)]) / 4
    B2 = sum([yavg[i] * X2[i] for i in range(4)]) / 4
    B3 = sum([yavg[i] * X3[i] for i in range(4)]) / 4

    btemp = [B0, B1, B2, B3]

    t = [abs(b) / Sbs for b in btemp]

    x3temp = [X3min, X3max, X3max, X3min]
    yr = [b0 + b3 * x for x in x3temp]
    yr1 = yr[0]
    yr2 = yr[1]
    yr3 = yr[2]
    yr4 = yr[3]

    S2ad = (3 / 2) * sum([(yr[i] - yavg[i]) ** 2 for i in range(4)])
    Fp = S2ad / Sb

    return render(request, "lab3.html", {
        "X1min": X1min,
        "X1max": X1max,
        "X2min": X2min,
        "X2max": X2max,
        "X3min": X3min,
        "X3max": X3max,
        "Xmaxavg": Xmaxavg,
        "Xminavg": Xminavg,
        "Ymin": Ymin,
        "Ymax": Ymax,
        "Y1": Y1,
        "Y2": Y2,
        "Y3": Y3,
        "Y4": Y4,
        "mY1": mY1,
        "mY2": mY2,
        "mY3": mY3,
        "mY4": mY4,
        "Yavg": Yavg,
        "mx1": mx1,
        "mx2": mx2,
        "mx3": mx3,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a11": a11,
        "a22": a22,
        "a33": a33,
        "a12": a12,
        "a21": a21,
        "a13": a13,
        "a31": a31,
        "a32": a32,
        "a23": a23,
        "b": b,
        "b0": b0,
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "test1": test1,
        "test2": test2,
        "test3": test3,
        "test4": test4,
        "X0": X0,
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "y1": y1,
        "y2": y2,
        "y3": y3,
        "y4": y4,
        "dy1": dy1,
        "dy2": dy2,
        "dy3": dy3,
        "dy4": dy4,
        "Gp": Gp,
        "Sb": Sb,
        "S2bs": S2bs,
        "Sbs": Sbs,
        "B": btemp,
        "t": t,
        "yr": yr,
        "yr1": yr1,
        "yr2": yr2,
        "yr3": yr3,
        "yr4": yr4,
        "S2ad": S2ad,
        "Fp": Fp,
    })
