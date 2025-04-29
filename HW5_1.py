import math

# Differential equation: y' = 1 + (y/t) + (y/t)^2
def f(t, y):
    yt = y / t
    return 1 + yt + yt**2

# Exact solution: y(t) = t * tan(ln t)
def exact_solution(t):
    return t * math.tan(math.log(t))

# Derivatives for Taylor method
def dfdt(t, y):
    return -y / (t**2) - 2 * y**2 / (t**3)

def dfdy(t, y):
    return 1 / t + 2 * y / (t**2)

# Common settings
h = 0.1
steps = int((2.0 - 1.0) / h)

# -------------------------------
# (a) Euler's Method
# -------------------------------
print("\n(a) Euler’s Method (h = 0.1)")
print(f"{'t':<5}{'Euler_y':>12}{'Exact_y':>12}{'Error':>12}")
print("-" * 41)

t = 1.0
y_euler = 0.0

for _ in range(steps + 1):
    y_exact = exact_solution(t)
    error = abs(y_euler - y_exact)
    print(f"{t:<5.2f}{y_euler:>12.6f}{y_exact:>12.6f}{error:>12.6f}")
    y_euler += h * f(t, y_euler)
    t += h

# -------------------------------
# (b) Taylor Method of Order 2
# -------------------------------
print("\n(b) Taylor’s Method of Order 2 (h = 0.1)")
print(f"{'t':<5}{'Taylor_y':>12}{'Exact_y':>12}{'Error':>12}")
print("-" * 41)

t = 1.0
y_taylor = 0.0

for _ in range(steps + 1):
    y_exact = exact_solution(t)
    error = abs(y_taylor - y_exact)
    print(f"{t:<5.2f}{y_taylor:>12.6f}{y_exact:>12.6f}{error:>12.6f}")
    f_val = f(t, y_taylor)
    df_total = dfdt(t, y_taylor) + dfdy(t, y_taylor) * f_val
    y_taylor += h * f_val + (h**2 / 2) * df_total
    t += h
