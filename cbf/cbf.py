import numpy as np
from cvxopt import matrix
from cvxopt import solvers

#Build barrier function model
def CBF(env, T_min, T_max, eta_1 = 0.5, eta_2 = 0.5):


    P = matrix(np.diag([0.0, 1e24,1e24]), tc='d')
    q = matrix(np.zeros(3))


    T = env.state[0]
    Q = env.state[1]
    a = np.delete(env.A[0], [1,2])
    a_1 = env.A[0][2]
    var = np.append(T, env.d[env.count_steps,1:]).T
    #print("var: {}, a: {}".format(var,a))
    temp = np.matmul(a, var)

    Delta_1 = -T_min + (eta_1 - 1)*(T - T_min) + temp
    Delta_2 = T_max + (eta_2 - 1)*(T_max - T) - temp
    
    G = np.array([[-a_1, -1., 0.], [a_1, 0., -1.], [-1., 0., 0.], [1., 0., 0.]]).astype(np.double)
    G = matrix(G,tc='d')

    h = np.array([Delta_1, Delta_2, -T_min, T_max])
    #print(h)
    h = np.squeeze(h).astype(np.double)
    h = matrix(h,tc='d')

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)

    u_bar = sol['x']
    # if np.abs(u_bar[1]) > 0.001:
    #     print("Violation of Safety: ")
    #     print(u_bar[1])
    return u_bar[0]


#===================================
#     CBF for RL
#===================================
def CBF_rl(env, T_rl, args, eta_1 = 0.5, eta_2 = 0.5):
    
    #print('running - CBF_rl')
    P = matrix(np.diag([1.0, 1e24,1e24]), tc='d')
    q = matrix(np.zeros(3))

    T_min = args['T_max_min'][0] 
    T_max=args['T_max_min'][1]
    T_set_min =args['T_set_max_min'][0] 
    T_set_max=args['T_set_max_min'][1]
    T = env.state[0]
    Q = env.state[1]
    a = np.delete(env.A[0], [1])
    a_1 = env.A[0][2]
    var = np.append([T, T_rl], env.d[env.count_steps,1:]).T
    #print("var: {}, a: {}".format(var,a))
    temp = np.matmul(a, var)

    Delta_1 = -T_min + (eta_1 - 1)*(T - T_min) + temp
    Delta_2 = T_max + (eta_2 - 1)*(T_max - T) - temp
    
    G = np.array([[-a_1, -1., 0.], [a_1, 0., -1.], [-1., 0., 0.], [1., 0., 0.]]).astype(np.double)
    G = matrix(G,tc='d')

    h = np.array([Delta_1, Delta_2, T_rl-T_set_min, T_set_max-T_rl])
    #print(h)
    h = np.squeeze(h).astype(np.double)
    h = matrix(h, tc='d')

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)

    u_bar = sol['x']
    # if np.abs(u_bar[1]) > 0.001:
    #     print("Violation of Safety: ")
    #     print(u_bar[1])
    return u_bar[0]


