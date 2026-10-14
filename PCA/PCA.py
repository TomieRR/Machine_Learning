import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

#srednia
srednia_X = np.mean(X, axis=0)
#centrowanie
Wycentrowane_X = X - srednia_X

#macierz kowarianji

n = Wycentrowane_X.shape[0]

Macierz_kowariancji = (Wycentrowane_X.T @ Wycentrowane_X ) / (n-1)

#wartosci i wektory wlasne

eigenvalues, eigenvectors = np.linalg.eigh(Macierz_kowariancji)

#sortowanie malejaco
kolejnosc = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[kolejnosc]
eigenvectors = eigenvectors[:, kolejnosc]

#nowe dwie najwazniejsze skladowe

W_2 = eigenvectors[:, :2]

components_df = pd.DataFrame(W_2, index=iris.feature_names, columns=["PC1", "PC2"])
print(components_df)

X_pca_2 = Wycentrowane_X @ W_2

#redukcja wymiarow z 4 do 3

W_3 = eigenvectors[:, :3]
X_pca_3 = Wycentrowane_X @ W_3

#Wagi wartości (ratio)

Wariancja_Ratio = eigenvalues/eigenvalues.sum()
Suma_Wariancja_Ratio = np.cumsum(Wariancja_Ratio)

#Udział zmienności
Wariancja_Ratio_2D = Suma_Wariancja_Ratio[:2].sum()
Wariancja_Ratio_3D = Suma_Wariancja_Ratio[:3].sum()

# wyprintowanie procentu dokladnosci w PC1 PC2
print("Udział zmienności zachowany w 2D: ",round(Wariancja_Ratio_2D, 4))
print("Udział zmienności zachowany w 3D: ",round(Wariancja_Ratio_3D, 4))


#Wizualizacja

#Wizualizacja 2D

plt.figure(figsize=(10,6))

for class_id, class_name in enumerate(iris.target_names):
    mask = y == class_id
    plt.scatter(X_pca_2[mask, 0], X_pca_2[mask, 1], label=class_name)


plt.grid(True)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris dla dwóch wymiarów po PCA")
plt.legend()
plt.show()

#Wizualizacja 3D

fig_3D = plt.figure()
ax = fig_3D.add_subplot(111, projection='3d')

for class_id, class_name in enumerate(iris.target_names):
    mask = y == class_id

    ax.scatter(
        X_pca_3[mask, 0],
        X_pca_3[mask, 1],
        X_pca_3[mask, 2],
        label=class_name
    )

plt.grid(True)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Iris dla 3D po PCA")
plt.legend()
plt.show()
