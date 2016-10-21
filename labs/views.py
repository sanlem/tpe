from django.shortcuts import render
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
    my = sum([row.Yavg for row in norm_table]) / len(norm_table)

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
    b0 = det([[my, mx1, mx2], [a11, a1, a2], [a22, a2, a3]]) / denominator
    b1 = det([[1, my, mx2], [mx1, a11, a2], [mx2, a22, a3]]) / denominator
    b2 = det([[1, mx1, my], [mx1, a1, a11], [mx2, a2, a22]]) / denominator

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
        'my': my,
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
