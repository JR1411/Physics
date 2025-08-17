import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

data = pd.read_csv("Lstm.csv", delimiter=';')

voltage = data["Volt"]
del_volt = data["err_x"] * voltage

pwr = data["PUI"]
pwr2 = data["Watt"]
del_pow = data["err_P"]

x_plot = np.linspace(voltage.min(), voltage.max(), 1000)


def model(x, R):
    return x**2 / R


popt, pcov = curve_fit(model, voltage, (pwr + pwr2) / 2, 730)

R = popt[0]
del_R = np.diag(np.sqrt(pcov))[0]
fit = model(x_plot, R)

pwr_hat = interp1d(voltage, pwr, kind='cubic')
pwr2_hat = interp1d(voltage, pwr2, kind='cubic')

error_PUI = abs(fit - pwr_hat(x_plot))
error_P = abs(fit - pwr2_hat(x_plot))

fig, ax = plt.subplots(2, 1)

ax[0].errorbar(voltage, pwr, del_volt, del_pow, fmt='o',
               color="orange", label=r"$P_{UI}$")
ax[0].errorbar(voltage, pwr2, del_volt, del_pow,
               fmt='o', color="purple", label='P')
ax[0].plot(x_plot, fit, 'r--', label='theory')
ax[0].legend()
ax[0].set(xlabel="voltage [V]", ylabel="power [P]",
          title=f"Resistance R = ({R:.1f}$\pm${del_R:.1f}) $\Omega$")
ax[0].grid()
ax[1].plot(x_plot, error_PUI, color="orange", label="error_PUI")
ax[1].plot(x_plot, error_P, color="purple", label="error_P")
ax[1].legend()
ax[1].set(xlabel="voltage [V]", ylabel="error [W]",
          title="absolute error comparison")
ax[1].grid()
plt.tight_layout()
plt.show()
