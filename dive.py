import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def dvdt(t, S, m, k, g):
    vx = S[0]
    vy = S[1]
    mag = np.sqrt(vx**2 + vy**2)
    gamma = k / m
    return [
        -gamma * mag * vx,
        -g - gamma * mag * vy
    ]


time = np.linspace(0, 50, 100)
m = 80
g = 9.81
vt = -55
k = m*g / vt**2
vx0 = 50
vy0 = 0

sol = solve_ivp(dvdt, (0, 50), (vx0, vy0), method='RK45',
                args=(m, k, g), t_eval=time)

vx = sol.y[0]
vy = sol.y[1]

idx = abs(vt - vy) / abs(vy) < 0.01
ans = time[idx]

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(time, vy / vt, color="blue", label=r"$\frac{v_y}{v_t}$")
ax.plot(ans[0], 0.99, marker='x', color='red', label='99% mark')
ax.legend()
ax.set(xlabel='time[s]', ylabel=r"$\frac{v_y}{v_t}$",
       title=f"99 % of terminal velocity after {ans[0]:.2f} s")
ax.grid()
plt.show()
