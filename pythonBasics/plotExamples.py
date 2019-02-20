import matplotlib.pyplot as plt
import numpy as np
import random as rnd

t = np.linspace(start=-10, stop=10, num=1e2)
noise = [5*(2* rnd.random() -1) for i in t]
y = t**2 + noise
approxPolynom = np.polyfit(t,y, deg=2)
yApprox = np.array([np.poly1d(approxPolynom)(i) for i in t])
lineMeasurement, lineApprox = plt.plot(t,y, t, yApprox)

plt.show()


plt.clf()

plt.hist(noise)
plt.show()


