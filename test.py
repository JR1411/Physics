import numpy as np 
import matplotlib.pyplot as plt 


x = np.linspace(0,1)

y = np.exp(x) 

plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.show()