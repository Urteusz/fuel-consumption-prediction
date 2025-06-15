import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error


def fuel_consumption_prediction(origin_option=0, split=0.2, random=42, test_depth=False, max_depth=None):
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

    if origin_option == 0:
        # Usuń kolumnę 'origin'
        df = df.drop('origin', axis=1)
        origin_desc = "bez kolumny 'origin'"
    elif origin_option == 1:
        # Zostaw kolumnę 'origin' jako numeryczną (1, 2, 3)
        origin_desc = "z kolumną 'origin' jako numeryczną"
    else:
        # One-Hot Encoding kolumny 'origin'
        df['origin'] = df['origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
        df = pd.get_dummies(df, columns=['origin'], prefix='', prefix_sep='')
        origin_desc = "z kolumną 'origin' jako One-Hot Encoding"

    X = df.drop('l_per_100km', axis=1)
    y = df['l_per_100km']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=random)

    print(f"\nRozmiar zbioru treningowego: {X_train.shape[0]} próbek")
    print(f"Rozmiar zbioru testowego: {X_test.shape[0]} próbek")

    if test_depth:
        # Testowanie różnych głębokości
        depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, None]
        mae_results = []

        print("\n--- Porównanie MAE dla różnych głębokości drzewa ---")
        for depth in depths:
            dt = DecisionTreeRegressor(max_depth=depth, random_state=random)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mae_results.append((str(depth), mae))
            print(f"max_depth = {depth}: MAE = {mae:.2f} L/100km")

            # Ważność cech dla każdej głębokości
            importances = dt.feature_importances_
            feature_importance_df = pd.DataFrame({'Cecha': X.columns, 'Ważność': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Ważność', ascending=False)

            print(f"\n--- Ważność Cech dla max_depth={depth} ---")
            print(feature_importance_df)
            print("----------------------")

        # Wykres porównawczy MAE
        depth_labels = [d[0] for d in mae_results]
        mae_values = [d[1] for d in mae_results]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=depth_labels, y=mae_values)
        plt.title('Porównanie MAE dla różnych głębokości drzewa decyzyjnego')
        plt.xlabel('Maksymalna głębokość drzewa')
        plt.ylabel('MAE [L/100km]')
        plt.show()

        print(f"Najmniejsze MAE: {min(mae_values):.2f} L/100km dla max_depth={depth_labels[mae_values.index(min(mae_values))]}")

    else:
        # Pojedynczy test z wybraną głębokością
        print(f"\n--- Test pojedynczego modelu z max_depth={max_depth} ---")

        # Model z wybraną głębokością
        final_tree = DecisionTreeRegressor(max_depth=max_depth, random_state=random)
        final_tree.fit(X_train, y_train)

        # Predykcja i ocena
        y_pred = final_tree.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"MAE = {mae:.2f} L/100km")

        # Ważność cech
        importances = final_tree.feature_importances_
        feature_importance_df = pd.DataFrame({'Cecha': X.columns, 'Ważność': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Ważność', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Ważność', y='Cecha', data=feature_importance_df)
        plt.title(f'Ważność cech w modelu predykcji spalania - max_depth={max_depth} ({origin_desc})', fontsize=16)
        plt.xlabel('Ważność')
        plt.ylabel('Cecha')
        plt.show()

        print("\n--- Ważność Cech ---")
        print(feature_importance_df)
        print("----------------------")

        # Wizualizacja uproszczonego drzewa (zawsze max_depth=3 dla czytelności)
        plt.figure(figsize=(20, 10))
        simple_tree = DecisionTreeRegressor(max_depth=3, random_state=random)
        simple_tree.fit(X_train, y_train)
        plot_tree(simple_tree, feature_names=X.columns, filled=True,
                  rounded=True, fontsize=10, precision=2)
        plt.title("Wizualizacja drzewa decyzyjnego (max_depth=3)", fontsize=16)
        plt.show()



def main():
    print("Prognozowanie spalania paliwa w samochodach (L/100km)")
    print("---------------------------------------------------")
    print("1. Usuń kolumnę 'origin'")
    print("2. Zostaw kolumnę 'origin' jako numeryczną (1=USA, 2=Europe, 3=Japan)")
    print("3. Użyj kodowania zmiennej 'origin' (One-Hot Encoding)")

    choice = input("Wybierz opcję (1, 2 lub 3): ")
    if choice not in ['1', '2', '3']:
        print("Nieprawidłowy wybór. Używam domyślnej opcji 1.")
        choice = '1'

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

    test_depth = input("Czy chcesz przetestować różne głębokości drzewa? (tak/nie, domyślnie nie): ").strip().lower()

    if test_depth == 'tak':
        # Testowanie różnych głębokości
        fuel_consumption_prediction(origin_option=int(choice) - 1, split=split, random=random, test_depth=True)
    else:
        # Pojedynczy test
        max_depth_input = input("Podaj maksymalną głębokość drzewa (domyślnie None - bez ograniczeń): ")
        if max_depth_input == '':
            max_depth = None
        else:
            max_depth = int(max_depth_input)

        fuel_consumption_prediction(origin_option=int(choice) - 1, split=split, random=random,
                                    test_depth=False, max_depth=max_depth)


if __name__ == "__main__":
    main()