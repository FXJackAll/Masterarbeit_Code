import matplotlib.pyplot as plt
import numpy as np

save_pic_path = "../figures/2D_visualization_comparison_restart_strategies_hom_hpb_aco_r_vs_mv.png"

x_labels = np.array(["(2, 11)", "(2, 21)", "(2, 51)",
                     "(5, 11)", "(5, 21)", "(5, 51)",
                     "(10, 11)", "(10, 21)", "(10, 51)"])

y_points_cigar_hom_f0 = np.array([1.0, 1.1, 1.1, 1.6, 1.4, 1.4, 1.5, 1.9, 2.0])
y_points_cigar_hom_adapt = np.array([1.4, 1.4, 1.5, 32.6, 26.2, 29.3])

y_points_ellipsoid_hom_f0 = np.array([1.0, 1.0, 1.0, 1.0, 2.4, 1.1, 1.1, 1.1, 1.1])
y_points_ellipsoid_hom_adapt = np.array([1.5, 1.5, 1.4, 13.0, 13.0, 15.1, 11.4, 8.6])

y_points_paraboloid_hom_f0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.0, 1.1])
y_points_paraboloid_hom_adapt = np.array([1.2, 1.4, 1.5, 8.8, 11.7, 10.6, 28.7, 33.4, 25.6])

y_points_tablet_hom_f0 = np.array([1.5, 1.2, 1.0, 1.8, 1.1, 1.1, 1.0, 1.0, 1.3])
y_points_tablet_hom_adapt = np.array([2.1, 2.0, 1.5, 21.4, 20.6, 26.3, 2.7])

plt.title("Hom-HPB-ACO$_\mathbb{R}$-VS-MV")

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

plt.legend()

plt.savefig(save_pic_path)
plt.show()