import numpy as np
import pandas as pd

def f_system(t, u):
    u1, u2 = u
    du1 = 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)
    du2 = -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)
    return np.array([du1, du2])

def exact_system(t):
    u1 = 2 * np.exp(-3*t) - np.exp(-39*t) + (1/3) * np.cos(t)
    u2 = -np.exp(-3*t) + 2 * np.exp(-39*t) - (1/3) * np.cos(t)
    return np.array([u1, u2])

def integrate_rk4(rhs, exact_fun, u0, h, T):
    n_steps = int(T / h)
    t_list = np.linspace(0, T, n_steps+1)
    u_vals = np.zeros((n_steps+1, len(u0)))
    exact_vals = np.zeros_like(u_vals)

    u = np.array(u0, dtype=float)
    u_vals[0] = u
    exact_vals[0] = exact_fun(0)

    for i in range(n_steps):
        t = t_list[i]
        k1 = rhs(t, u)
        k2 = rhs(t + h/2, u + h/2 * k1)
        k3 = rhs(t + h/2, u + h/2 * k2)
        k4 = rhs(t + h,     u + h   * k3)
        u = u + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

        u_vals[i+1] = u
        exact_vals[i+1] = exact_fun(t_list[i+1])

    return pd.DataFrame({
        't':         t_list,
        'u1_RK4':    u_vals[:,0],
        'u1_exact':  exact_vals[:,0],
        'u1_error':  np.abs(u_vals[:,0]  - exact_vals[:,0]),
        'u2_RK4':    u_vals[:,1],
        'u2_exact':  exact_vals[:,1],
        'u2_error':  np.abs(u_vals[:,1]  - exact_vals[:,1]),
    })

pd.set_option('display.float_format', '{:>12.6f}'.format)
pd.set_option('display.expand_frame_repr', False)

u0 = [4/3, 2/3]
T  = 1.0

print("\nResults for h = 0.05\n")
df_h005 = integrate_rk4(f_system, exact_system, u0, 0.05, T)
print(df_h005.to_string(index=False))

print("\nResults for h = 0.1\n")
df_h01 = integrate_rk4(f_system, exact_system, u0, 0.1, T)
print(df_h01.to_string(index=False))

