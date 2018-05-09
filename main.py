import numpy as np

def evaluate(result):
    return (1 + (result[0] + result[1] + 1)**2 * (19-14*result[0]+3*result[0]*result[0]-14*result[1]+6*result[0]*result[1]+3*result[1]*result[1])) * \
           (30 + (2*result[0] - 3*result[1])**2 * (18-32*result[0]+12*result[0]*result[0]+48*result[1]-36*result[0]*result[1]+27*result[1]*result[1]))

def loading(n):
    loading_char = ['|', '/', '-', '\\']
    return loading_char[n]

def new_position(particles, velocity):
    X_MIN, X_MAX = -2, 2
    Y_MIN, Y_MAX = -2, 2
    particles += velocity
    particles = np.transpose(np.clip(np.transpose(particles), [X_MIN, Y_MIN], [X_MAX, Y_MAX]))
    return particles

def matrix_pso():
    n_particles = 100
    max_iterations = 10000
    dims = 2
    particles = np.random.uniform(-2, 2, (dims, n_particles))
    velocities = np.random.uniform(-4, 4, (dims, n_particles))
    p_best = np.array(particles[:])
    g_best = np.array(particles[:, np.argmin(evaluate(particles)), None])

    for k in range(max_iterations):
        if np.any(g_best > 512):
            print('{}: {}'.format(k, g_best))
            return
        print('{} GLOBAL: {:.3f}\r'.format(loading(k%4), evaluate(g_best)[0]), end='')
        c1 = (4-1)*(k/max_iterations)+1
        c2 = (4-1)*(k/max_iterations)+1
        gamma = (0.9-0.4)*((max_iterations-k)/max_iterations)+0.4
        rands = np.random.rand(2, n_particles)
        velocities = (gamma * velocities) + (c1 * rands[0] * (p_best - particles)) \
                          + (c2 * rands[1] * (-particles + g_best))
        particles = new_position(particles, velocities)
        new_values = evaluate(particles)

        mask = new_values < evaluate(p_best)
        for idx in range(n_particles):
            p_best[:, idx] = np.array(particles[:, idx]) if mask[idx] else np.array(p_best[:, idx])

        min_particle_idx = np.argmin(new_values)
        if evaluate(g_best) > new_values[min_particle_idx]:
            g_best = np.array(particles[:, min_particle_idx, None])
    print('Optimal value: ({:.6f},{:.6f}) @ {:.4f}'.format(g_best[0][0], g_best[1][0], evaluate(g_best)[0]))
    return evaluate(g_best)[0]

res = matrix_pso()
true_res = evaluate((0, -1))
print('Found solution is less than optimal? {:.10f} < {:.10f} : {}'.format(res, true_res, res < true_res))

