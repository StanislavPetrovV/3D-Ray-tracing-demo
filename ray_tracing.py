from ray_tracing_functions import *


obj0 = ('plane', 0.0, -1.1, 0.0, 0.0, 0.25)
obj1 = ('sphere', 0.0, -0.3, -3.5, 0.8, 0.7)
obj2 = ('sphere', 0.7, -0.8, -2.7, 0.3, 0.5)
obj3 = ('sphere', -0.8, -0.9, -2.7, 0.2, 0.6)
obj4 = ('sphere', 0.2, -1.02, -2.3, 0.08, 0.6)
obj5 = ('sphere', -4.2, 0.99, -4.3, 2.1, 0.6)
obj6 = ('sphere', 4.2, 2.8, -6.3, 4.1, 0.75)
scene = (obj0, obj1, obj2, obj3, obj4, obj5, obj6)
spec_color = (0.1, 0.1, 0.1)
light_pos = (-3.0, 15.0, 2.5)
origin = (0.0, 0.0, 0.0)


@njit(fastmath=True)#, cache=True)
def ray_tracing(key):
    SCALE = key[5]
    REAL_WIDTH = WIDTH // SCALE
    REAL_HEIGHT = HEIGHT // SCALE
    HALF_REAL_WIDTH = REAL_WIDTH // 2
    HALF_REAL_HEIGHT = REAL_HEIGHT // 2
    CAM_DIST = -HALF_REAL_HEIGHT / TAN_A
    res = []
    depth_max = 5
    for x in range(REAL_WIDTH):
        for y in range(REAL_HEIGHT):
            rd = normalize_vec3((x - HALF_REAL_WIDTH, HALF_REAL_HEIGHT - y, CAM_DIST))
            ro = (key[1], key[2], key[0])

            col = (0.0, 0.0, 0.0)
            reflection = 1
            # reflection steps
            for depth in range(depth_max):
                t = np.inf
                for i,obj in enumerate(scene):
                    t_obj, p_obj, n_obj = intersect(ro, rd, obj[:-1])
                    if 0 < t_obj < t:
                        t, p, N, obj_ind = t_obj, p_obj, n_obj, i
                if t == np.inf:
                    break

                # find if the point is shadowed or not
                L = normalize_vec3(sub_vecs3(light_pos, p))
                t_sh = np.inf
                for i, obj in enumerate(scene):
                    if i != obj_ind:
                        t_obj, _, _ = intersect(sum_vecs3(p, mul_vec3_n(N, 0.0001)), L, obj[:-1])
                        if 0 < t_obj < t_sh:
                            t_sh = t_obj
                if t_sh < np.inf:
                    break

                # shading
                color = get_color(obj_ind, p)
                color = div_vec3_n(color, 225)
                dot_NL = max(dot_vecs3(N, L), 0)
                color = mul_vec3_n(color, max(dot_NL + 0.1, 0) / (1.0 + 0.1))

                # reflection ray
                ro = sum_vecs3(p, mul_vec3_n(N, 0.0001))
                rd = normalize_vec3( sub_vecs3( rd, mul_vec3_n( N, 2 * dot_vecs3(rd, N) ) ) )
                col = sum_vecs3(col, mul_vec3_n(color, reflection))
                reflection *= scene[obj_ind][-1]

            col = mul_vec3_n(clip_vec3(col, 0, 1), 255)
            res.append( (col, (int(x * SCALE), int(y * SCALE))) )
    return res



