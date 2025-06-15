import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def fuel_consumption_prediction(is_origin=0, split=0.2, random=42):
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)

    # Wczytanie danych
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                    'acceleration', 'model_year', 'origin', 'car_name']
    url = 'auto-mpg.data'
    df = pd.read_csv(url, names=column_names, na_values='?', comment='\t',
                     sep=' ', skipinitialspace=True)

    df['l_per_100km'] = 235.214583 / df['mpg']
    df['weight'] = df['weight'] * 0.453592
    df = df.drop(['mpg', 'car_name'], axis=1)
    df = df.dropna(subset=['horsepower'])

    if is_origin == 1:
        df['origin'] = df['origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
        df = pd.get_dummies(df, columns=['origin'], prefix='', prefix_sep='')
    else:
        df = df.drop('origin', axis=1)

    X = df.drop('l_per_100km', axis=1)
    y = df['l_per_100km']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=random)

    print(f"\nRozmiar zbioru treningowego: {X_train.shape[0]} próbek")
    print(f"Rozmiar zbioru testowego: {X_test.shape[0]} próbek")

    # Testowane głębokości
    depths = [1, 2, 3, 5, 10, None]
    mae_results = []

    print("\n--- Porównanie MAE dla różnych głębokości drzewa ---")
    for depth in depths:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=random)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mae_results.append((str(depth), mae))
        print(f"max_depth = {depth}: MAE = {mae:.2f} L/100km")
    print("-----------------------------------------------------")

    # Wykres porównawczy MAE
    depth_labels = [d[0] for d in mae_results]
    mae_values = [d[1] for d in mae_results]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=depth_labels, y=mae_values)
    plt.title('Porównanie MAE dla różnych głębokości drzewa decyzyjnego')
    plt.xlabel('Maksymalna głębokość drzewa')
    plt.ylabel('MAE [L/100km]')
    plt.show()

    # Wizualizacja uproszczonego drzewa
    plt.figure(figsize=(20, 10))
    simple_tree = DecisionTreeRegressor(max_depth=3, random_state=random)
    simple_tree.fit(X_train, y_train)
    plot_tree(simple_tree, feature_names=X.columns, filled=True,
              rounded=True, fontsize=10, precision=2)
    plt.title("Wizualizacja drzewa decyzyjnego (max_depth=3)", fontsize=16)
    plt.show()

    # Ważność cech
    final_tree = DecisionTreeRegressor(random_state=random)
    final_tree.fit(X_train, y_train)
    importances = final_tree.feature_importances_
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




def main():
    print("Prognozowanie spalania paliwa w samochodach (L/100km)")
    print("---------------------------------------------------")
    print("1. Usuń kolumnę 'origin'")
    print("2. Użyj kodowania zmiennej 'origin' (One-Hot Encoding)")
    choice = input("Wybierz opcję (1 lub 2): ")

    while choice not in ['1', '2']:
        print("Nieprawidłowy wybór. Proszę wybrać 1 lub 2.")
        choice = input("Wybierz opcję (1 lub 2): ")

    split = input("Podaj wartość podziału zbioru testowego (domyślnie 0.2): ")
    if split == '':
        split = 0.2
    else:
        split = float(split)
    random = input("Podaj wartość random_state (domyślnie 42): ")
    if random == '':
        random = 42
    else:
        random = int(random)
    fuel_consumption_prediction(is_origin=int(choice) - 1, split=split, random=random)

if __name__ == "__main__":
    main()