import math
import pandas as pd

# 定義微分方程
def f1(t, u1, u2):
    return 9*u1 + 24*u2 + 5*math.cos(t) - (1/3)*math.sin(t)

def f2(t, u1, u2):
    return -24*u1 - 52*u2 - 9*math.cos(t) + (1/3)*math.sin(t)

# 精確解
def u1_exact(t):
    return 2 * math.exp(-3 * t) - math.exp(-39 * t) + (1/3) * math.cos(t)

def u2_exact(t):
    return -math.exp(-3 * t) + 2 * math.exp(-39 * t) - (1/3) * math.cos(t)

# Runge-Kutta 方法 + pandas 表格輸出
def runge_kutta_system(h, t_end):
    t = 0.0
    u1 = 4/3
    u2 = 2/3

    rows = []

    while t <= t_end + 1e-8:
        u1_e = u1_exact(t)
        u2_e = u2_exact(t)
        err1 = abs(u1 - u1_e)
        err2 = abs(u2 - u2_e)

        rows.append({
            "t": round(t, 2),
            "u1_RK": u1,
            "u1_exact": u1_e,
            "err_u1": err1,
            "u2_RK": u2,    
            "u2_exact": u2_e,  
            "err_u2": err2
        })

        # Runge-Kutta 步驟
        k1_1 = h * f1(t, u1, u2)
        k1_2 = h * f2(t, u1, u2)

        k2_1 = h * f1(t + h/2, u1 + k1_1/2, u2 + k1_2/2)
        k2_2 = h * f2(t + h/2, u1 + k1_1/2, u2 + k1_2/2)

        k3_1 = h * f1(t + h/2, u1 + k2_1/2, u2 + k2_2/2)
        k3_2 = h * f2(t + h/2, u1 + k2_1/2, u2 + k2_2/2)

        k4_1 = h * f1(t + h, u1 + k3_1, u2 + k3_2)
        k4_2 = h * f2(t + h, u1 + k3_1, u2 + k3_2)

        u1 += (k1_1 + 2*k2_1 + 2*k3_1 + k4_1) / 6
        u2 += (k1_2 + 2*k2_2 + 2*k3_2 + k4_2) / 6
        t += h

    # 建立 DataFrame 並格式化輸出
    df = pd.DataFrame(rows)
    pd.set_option('display.float_format', '{:.10f}'.format)
    print(f"\nResults for h = {h}")
    print(df.to_string(index=False))

# 執行兩組步長
runge_kutta_system(h=0.05, t_end=1.0)
runge_kutta_system(h=0.1, t_end=1.0)

