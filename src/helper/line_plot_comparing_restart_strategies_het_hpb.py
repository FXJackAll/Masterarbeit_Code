from json.decoder import NaN

import matplotlib.pyplot as plt
import numpy as np

from src.helper.box_plot_comparing_algorithms import save_pic_path

save_pic_path = "../figures/2D_visualization_comparison_restart_strategies_het_hpb_aco_r_vs_mv.png"

x_labels = np.array(["(2, 11)", "(2, 21)", "(2, 51)",
                     "(5, 11)", "(5, 21)", "(5, 51)",
                     "(10, 11)", "(10, 21)", "(10, 51)"])

y_points_cigar_het_f0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
y_points_cigar_het_adapt = np.array([1.6, 1.5, 1.9, 26.4, 22.9, 27.4])

y_points_ellipsoid_het_f0 = np.array([1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
y_points_ellipsoid_het_adapt = np.array([1.8, 1.8, 1.6, 14.6, 15.6, 16.3, 5.7, 1.3, 9.2])

y_points_paraboloid_het_f0 = np.array([1.3, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0])
y_points_paraboloid_het_adapt = np.array([1.7, 1.7, 1.6, 9.9, 10.6, 13.3, 28.1, 35.2, 23.2])

y_points_tablet_het_f0 = np.array([1.0, 1.0, 5.8, 1.7, 1.0, 1.0, 1.4, 1.0, 1.3])
y_points_tablet_het_adapt = np.array([2.3, 2.0, 1.6, 17.1, 21.6, 15.2, None, 22.7, 2.7])

plt.title("Het-HPB-ACO$_\mathbb{R}$-VS-MV")

plt.xlabel("Kombination aus Dimension und Anzahl diskreter Punkte")
plt.ylabel("relative Anzahl an Funktionsauswertungen")

# plt.setp(title)

plt.plot(x_labels, y_points_cigar_het_f0, color='red', label="Cigar-f0")
plt.plot(y_points_ellipsoid_het_f0, color='tomato', label="Ellipsoid-f0")
plt.plot(y_points_paraboloid_het_f0, color='salmon', label="Paraboloid-f0")
plt.plot(y_points_tablet_het_f0, color='darksalmon', label="Tablet-f0")

plt.plot(y_points_cigar_het_adapt, color='navy', label="Cigar-adapt")
plt.plot(y_points_ellipsoid_het_adapt, color='darkblue', label="Ellipsoid-adapt")
plt.plot(y_points_paraboloid_het_adapt, color='blue', label="Paraboloid-adapt")
plt.plot(y_points_tablet_het_adapt, color='darkslateblue', label="Tablet-adapt")

plt.legend()

# plt.savefig(save_pic_path)
plt.show()