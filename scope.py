import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("scope_1.csv", skiprows=1)

time = data["second"]
current_in = data["Volt"]
current_out = data["Volt1"]

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(time, current_in, color='blue', label='incoming currrent')
ax.plot(time, current_out, color="red", label='outgoing current')
ax.grid()
ax.set(xlabel="time [s]", ylabel="current [A]")
ax.legend()
plt.show()
