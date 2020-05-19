import math
from numba import njit

@njit(fastmath=True, cache=True)
def mod_vec3_n(vec, n):
    return (vec[0] % n, vec[1] % n, vec[2] % n)


@njit(fastmath=True, cache=True)
def length_vec3(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


@njit(fastmath=True, cache=True)
def length_vec2(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)


@njit(fastmath=True, cache=True)
def sub_vecs3(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


@njit(fastmath=True, cache=True)
def sub_vecs2(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1])


@njit(fastmath=True, cache=True)
def sub_vec3_n(v1, n):
    return (v1[0] - n, v1[1] - n, v1[2] - n)


@njit(fastmath=True, cache=True)
def sub_n_vec3(n, v1):
    return (-v1[0] + n, -v1[1] + n, -v1[2] + n)


@njit(fastmath=True, cache=True)
def sub_vec2_n(v1, n):
    return (v1[0] - n, v1[1] - n)


@njit(fastmath=True, cache=True)
def sum_vecs3(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])


@njit(fastmath=True, cache=True)
def sum_vec2_n(v1, n):
    return (v1[0] + n, v1[1] + n)


@njit(fastmath=True, cache=True)
def mul_vec3_n(v1, n):
    return (v1[0] * n, v1[1] * n, v1[2] * n)


@njit(fastmath=True, cache=True)
def mul_vec2_n(v1, n):
    return (v1[0] * n, v1[1] * n)


@njit(fastmath=True, cache=True)
def div_vec3_n(v1, n):
    n = 1 / n
    return (v1[0] * n, v1[1] * n, v1[2] * n)


@njit(fastmath=True, cache=True)
def div_vec2_n(v1, n):
    n = 1 / n
    return (v1[0] * n, v1[1] * n)


@njit(fastmath=True, cache=True)
def div_vecs2(v1, v2):
    v2 = 1 / v2
    return (v1[0] * v2[0], v1[1] * v2[1])


@njit(fastmath=True, cache=True)
def dot_vecs3(v1, v2):
    return (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])


@njit(fastmath=True, cache=True)
def abs_vec3(vec):
    return (abs(vec[0]), abs(vec[1]), abs(vec[2]))


@njit(fastmath=True, cache=True)
def normalize_vec3(vec):
    len_vec = 1 / math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    return (vec[0] * len_vec, vec[1] * len_vec, vec[2] * len_vec)


@njit(fastmath=True, cache=True)
def normalize_vec2(vec):
    len_vec = 1 / math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return (vec[0] * len_vec, vec[1] * len_vec)


@njit(fastmath=True, cache=True)
def cross_vecs3(v1, v2):
    return (v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0])


@njit(fastmath=True, cache=True)
def clip_n(n, min, max):
    if n > max:
        return max
    if n < min:
        return min
    return n


@njit(fastmath=True, cache=True)
def clip_vec3(v, min, max):
    return clip_n(v[0], min, max), clip_n(v[1], min, max), clip_n(v[2], min, max)


@njit(fastmath=True, cache=True)
def view_world_matrix(eye, look_at):
    # eye = (0.0, 0.0, 0.0)
    # look_at = (0.0, 0.0, -1.0)
    up = (0.0, 1.0, 0.0)
    cw = normalize_vec3(sub_vecs3(eye, look_at))
    cu = cross_vecs3(up, cw)
    cv = cross_vecs3(cw, cu)
    cu = (cu[0], cu[1], cu[2], 0.0)
    cv = (cv[0], cv[1], cv[2], 0.0)
    cw = (cw[0], cw[1], cw[2], 0.0)
    return (cu, cv, cw)


@njit(fastmath=True)
def mul_matrix_vec3(a, b):
    c0 = (a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2] + a[0][3] * 1)
    c1 = (a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2] + a[1][3] * 1)
    c2 = (a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2] + a[2][3] * 1)
    return (c0, c1, c2)


@njit(fastmath=True, cache=True)
def rotate_y_matrix(ray_dir, angle):
    s = math.sin(angle)
    c = math.cos(angle)
    a0 = (c, 0, -s, 0)
    a1 = (0, 1, 0, 0)
    a2 = (s, 0, c, 0)
    a = (a0, a1, a2)
    return mul_matrix_vec3(a, ray_dir)


@njit(fastmath=True, cache=True)
def rotate_x_matrix(ray_dir, angle):
    s = math.sin(angle)
    c = math.cos(angle)
    a0 = (1, 0, 0, 0)
    a1 = (0, c, -s, 0)
    a2 = (0, s, c, 0)
    a = (a0, a1, a2)
    return mul_matrix_vec3(a, ray_dir)


@njit(fastmath=True, cache=True)
def rotate_z_matrix(ray_dir, angle):
    s = math.sin(angle)
    c = math.cos(angle)
    a0 = (c, -s, 0, 0)
    a1 = (s, c, 0, 0)
    a2 = (0, 0, 1, 0)
    a = (a0, a1, a2)
    return mul_matrix_vec3(a, ray_dir)


@njit(fastmath=True, cache=True)
def translation_matrix(ray_dir, vec):
    a0 = (1, 0, 0, vec[0])
    a1 = (0, 1, 0, vec[1])
    a2 = (0, 0, 1, vec[2])
    a = (a0, a1, a2)
    return mul_matrix_vec3(a, ray_dir)


@njit(fastmath=True, cache=True)
def scale_matrix(ray_dir, n):
    a0 = (n, 0, 0, 0)
    a1 = (0, n, 0, 0)
    a2 = (0, 0, n, 0)
    a = (a0, a1, a2)
    return mul_matrix_vec3(a, ray_dir)


@njit(fastmath=True, cache=True)
def rotate_y(vec3, angle):
    s = math.sin(angle)
    c = math.cos(angle)
    return (vec3[0] * s, vec3[1], vec3[2] * c)

@njit(fastmath=True, cache=True)
def rotate_x(vec3, angle):
    s = math.sin(angle)
    c = math.cos(angle)
    return (vec3[0], vec3[1] * s, vec3[2] * c)

@njit(fastmath=True, cache=True)
def rotate_z(vec3, angle):
    s = math.sin(angle)
    c = math.cos(angle)
    return (vec3[0] * s, vec3[1] * s, vec3[2])

