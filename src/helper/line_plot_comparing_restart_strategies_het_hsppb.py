import matplotlib.pyplot as plt
import numpy as np

from src.helper.box_plot_comparing_algorithms import save_pic_path

save_pic_path = "../figures/2D_visualization_comparison_restart_strategies_het_hsppb_aco_r_vs_mv.png"

x_labels = np.array(["(2, 11)", "(2, 21)", "(2, 51)",
                     "(5, 11)", "(5, 21)", "(5, 51)",
                     "(10, 11)", "(10, 21)", "(10, 51)"])

y_points_cigar_hom_f0 = np.array([1.2, 1.1, 1.6, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0])
y_points_cigar_hom_adapt = np.array([1.7, 1.7, 1.6, 22.8, 26.5, 26.8])

y_points_ellipsoid_hom_f0 = np.array([6.9, 2.5, 1.2, 1.2, 2.2, 1.1, 1.1, 1.1, 1.1])
y_points_ellipsoid_hom_adapt = np.array([2.6, 2.4, 2.5, 26.2, 31.5, 33.9])

y_points_paraboloid_hom_f0 = np.array([1.5, 1.2, 1.2, 1.2, 1.2, 1.3, 1.1, 1.2, 1.2])
y_points_paraboloid_hom_adapt = np.array([1.7, 2.4, 2.4, 31.2, 39.7, 45.9])

y_points_tablet_hom_f0 = np.array([1.5, 1.3, 1.0, 1.0, 1.2, 1.2, 1.2, 1.0, 1.0])
y_points_tablet_hom_adapt = np.array([4.1, 4.8, 2.9, 41.5, 36.7, 33.3])

plt.title("Het-HSPPB-ACO$_\mathbb{R}$-VS-MV")

plt.xlabel("Kombination aus Dimension und Anzahl diskreter Punkte")
plt.ylabel("relative Anzahl an Funktionsauswertungen")

# plt.setp(title)

plt.plot(x_labels, y_points_cigar_hom_f0, color='red', label="Cigar-f0")
plt.plot(y_points_ellipsoid_hom_f0, color='tomato', label="Ellipsoid-f0")
plt.plot(y_points_paraboloid_hom_f0, color='salmon', label="Paraboloid-f0")
plt.plot(y_points_tablet_hom_f0, color='darksalmon', label="Tablet-f0")

plt.plot(y_points_cigar_hom_adapt, color='navy', label="Cigar-adapt")
plt.plot(y_points_ellipsoid_hom_adapt, color='darkblue', label="Ellipsoid-adapt")
plt.plot(y_points_paraboloid_hom_adapt, color='blue', label="Paraboloid-adapt")
plt.plot(y_points_tablet_hom_adapt, color='darkslateblue', label="Tablet-adapt")

plt.legend(loc="upper left")

plt.savefig(save_pic_path)
plt.show()