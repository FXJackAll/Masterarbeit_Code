import matplotlib.pyplot as plt
import numpy as np

x_labels = np.array(["(2, 11)", "(2, 21)", "(2, 51)",
                     "(5, 11)", "(5, 21)", "(5, 51)",
                     "(10, 11)", "(10, 21)", "(10, 51)"])

y_points_hom_f0 = np.array([1.0, 1.1, 1.1, 1.6, 1.4, 1.4, 1.5, 1.9, 2.0])
y_points_hom_adapt = np.array([1.4, 1.4, 1.5, 32.6, 26.2, 29.3])

plt.title("Hom-HPB-ACO$_\mathbb{R}$-VS-MV")

plt.xlabel("Kombination aus Dimension und Anzahl diskreter Punkte")
plt.ylabel("relative Anzahl an Funktionsauswertungen")

# plt.setp(title)

plt.plot(x_labels, y_points_hom_f0, color='r', label="Cigar-f0")
plt.plot(y_points_hom_adapt, color='b', label="Cigar-adapt")

plt.legend()

plt.show()