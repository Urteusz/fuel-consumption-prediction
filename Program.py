import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ustawienia wizualizacji
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Wczytanie i wstępna analiza danych
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
url = 'auto-mpg.data'
df = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

# Wzór: L/100km = 235.214 / MPG
df['l_per_100km'] = 235.214583 / df['mpg']
df = df.drop('mpg', axis=1)
# Wzór: 1 funt = 0.453592 kg
df['weight'] = df['weight'] * 0.453592
print("\nDane po konwersji na L/100km i kilogramy:")
print(df.head())

# Usunięcie niepotrzebnej kolumny z nazwą samochodu
df = df.drop('car_name', axis=1)

# Usuwamy rekordy z brakującymi wartościami w kolumnie 'horsepower' tylko 6 rekordów
df = df.dropna(subset=['horsepower'])

# Usunięcie kolumny 'origin' daje lepsze wyniki niż jej kodowanie
df = df.drop('origin', axis=1)

# Kodowanie zmiennej kategorialnej 'origin' (One-Hot Encoding) Wyniki gorsze niż usunięcie kolumny 'origin'
# df['origin'] = df['origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
# df = pd.get_dummies(df, columns=['origin'], prefix='', prefix_sep='')

# Podział na zbiór cech (X) i cel (y)
# Teraz naszą zmienną docelową jest 'l_per_100km'
X = df.drop('l_per_100km', axis=1)
y = df['l_per_100km']

# Podział na zbiór treningowy i testowy (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nRozmiar zbioru treningowego: {X_train.shape[0]} próbek")
print(f"Rozmiar zbioru testowego: {X_test.shape[0]} próbek")

# Budowa i trenowanie modelu drzewa decyzyjnego
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)
print("\nModel drzewa decyzyjnego został pomyślnie wytrenowany.")

# Ewaluacja modelu
y_pred = dt_regressor.predict(X_test)

# Obliczenie metryk
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Metryki Ewaluacji Modelu (Cel: L/100km) ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} L/100km")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} L/100km")
print(f"Współczynnik R-squared (R²): {r2:.3f}")
print("-------------------------------------------------")
print("Interpretacja: Model myli się średnio o {:.2f} litra na 100 km.".format(mae))

# Wizualizacja i interpretacja

# Wizualizacja drzewa (wersja uproszczona dla czytelności)
plt.figure(figsize=(20, 10))
simple_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
simple_tree.fit(X_train, y_train)
# W liściach drzewa (value) będzie teraz przewidywane spalanie w L/100km
plot_tree(simple_tree, feature_names=X.columns, filled=True, rounded=True, fontsize=10, precision=2)
plt.title("Wizualizacja drzewa decyzyjnego (predykcja L/100km, max_depth=3)", fontsize=16)
plt.show()

# Ważność cech (Feature Importance)
importances = dt_regressor.feature_importances_
feature_importance_df = pd.DataFrame({'Cecha': X.columns, 'Ważność': importances})
feature_importance_df = feature_importance_df.sort_values(by='Ważność', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Ważność', y='Cecha', data=feature_importance_df)
plt.title('Ważność cech w modelu predykcji spalania (L/100km)', fontsize=16)
plt.xlabel('Ważność')
plt.ylabel('Cecha')
plt.show()

print("\n--- Ważność Cech ---")
print(feature_importance_df)
print("----------------------")
