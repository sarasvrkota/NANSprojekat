
from util import *
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Učitavanje podataka
df = pd.read_csv('./data/berlin_weekends.csv')

# Eliminisanje redova sa nedostajućim vrednostima
df = df.dropna()
df.isnull().sum()  # Provera nedostajućih vrednosti
df = df.dropna()   # Uklanjanje redova sa nedostajućim vrednostima
df.dtypes  # Provera tipova podataka
print(df.dtypes)


# Konverzija boolean vrednosti u numeričke (0 ili 1)
df['room_shared'] = df['room_shared'].astype('int64')
df['room_private'] = df['room_private'].astype('int64')
df['host_is_superhost'] = df['host_is_superhost'].astype('int64')

# Mapiranje kategoričkih vrednosti u numerički oblik
room_type_mapping = {'Private room': 0, 'Entire home/apt': 1, 'Shared room': 2}
df['room_type'] = df['room_type'].map(room_type_mapping) #string to int
print(df.dtypes)

# Uklanjanje outliers-a
columns_to_check = ['realSum', 'biz', 'metro_dist', 'host_is_superhost', 'guest_satisfaction_overall', 'Unnamed: 0', 'rest_index', 'room_private', 'multi', 'dist', 'lat', 'lng', 'attr_index_norm', 'attr_index']
df = remove_outliers_iqr(df, columns=columns_to_check)


x = df.drop(['realSum', 'biz', 'metro_dist', 'host_is_superhost', 'guest_satisfaction_overall', 'Unnamed: 0', 'rest_index','room_private', 'multi', 'dist', 'lat', 'lng', 'attr_index_norm', 'attr_index'], axis=1)
y = df['realSum']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)


model = get_fitted_model(x_train, y_train) #dobijanje linearnog regresionog modela, koristi metodu najmanjih kvadrata

r2 = get_rsquared_adj1(model, x_test, y_test)
print(r2)
print(model.summary())


print("\n****PRETPOSTAVKE LINEARNE REGRESIJE****\n")

#LINEARNOST
x_with_const = sm.add_constant(x)
print('1. LINEARNOST: ', end='')
is_linearity_found, p_value = linear_assumption(model, x_with_const, y, plot=False)
if is_linearity_found: 
    print('Veza jeste linearna. Pretpostavka o linearnoj vezi ispunjena.')
else: 
    print('Nije ispunjena pretpostavka linearnosti.')


#NEZAVISNOST GREŠAKA
autocorrelation, dw_value = independence_of_errors_assumption(model, x_with_const, y, plot=False)
print("2. NEZAVISNOST GREŠAKA:")
print(f"\tTip autocorelacije: {autocorrelation}")
print(f"\tDurbin-Watson statistika: {dw_value}")  


#NORMALNOST GREŠAKA
n_dist_type, p_value = normality_of_errors_assumption(model, x_with_const, y, plot=False)
if n_dist_type == 'normal':
    print('3. NORMALNOST GREŠAKA: Greške su normalno rasporedjene')
else:
    print('3. NORMALNOST GREŠAKA: Greške nisu normalno rasporedjene')


#JEDNAKA VARIJANSA GREŠAKA   
# Provera pretpostavke homoskedastičnosti
dist_type, p_value = equal_variance_assumption(model, x_with_const, y, plot=False)
if dist_type == 'equal' and p_value >= 0.05:
    print('4. JEDNAKA VARIJANSA GREŠAKA: Pretpostavka homoskedastičnosti je zadovoljena.')
else:
    print('4. JEDNAKA VARIJANSA GREŠAKA: Pretpostavka homoskedastičnosti nije zadovoljena.')
    print(f"Tip raspodele grešaka: {dist_type}")
    print(f"P-vrednost za test homoskedastičnosti: {p_value}")


####################################################
from sklearn.linear_model import Lasso

# Inicijalizacija Lasso modela
lasso_model = Lasso(alpha=0.1)  # Podešavanje parametra alpha

# Treniranje Lasso modela na trening skupu
lasso_model.fit(x_train, y_train)

# Predviđanje na test skupu
y_pred_lasso = lasso_model.predict(x_test)

# Evaluacija Lasso modela
rmse_lasso = mean_squared_error(y_test, y_pred_lasso, squared=False)
r2_lasso = r2_score(y_test, y_pred_lasso)


# Ispis rezultata evaluacije Lasso modela

print(f"R-squared Lasso Regression: {r2_lasso}")

####################################################

from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

svr_rbf.fit(x_train, y_train)

y_pred_svr_rbf = svr_rbf.predict(x_test)


rmse_svr_rbf = mean_squared_error(y_test, y_pred_svr_rbf, squared=False)
r2_svr_rbf = r2_score(y_test, y_pred_svr_rbf)


print(f"R-squared SVM-RBF: {r2_svr_rbf}")


####################################################
from sklearn.ensemble import GradientBoostingRegressor

# Inicijalizacija Gradient Boosting modela
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Treniranje modela na trening skupu
gb_model.fit(x_train, y_train)

# Predviđanje na test skupu
y_pred_gb = gb_model.predict(x_test)

# Evaluacija Gradient Boosting modela
rmse_gb = mean_squared_error(y_test, y_pred_gb, squared=False)
r2_gb = r2_score(y_test, y_pred_gb)

# Ispis rezultata evaluacije Gradient Boosting modela
print(f"R-squared Gradient Boosting: {r2_gb}")

#####################################################


# Prikazivanje svih jedinstvenih vrednosti u određenoj koloni
'''unique_values = df['room_type'].unique()
print(unique_values) 
unique_values = df['host_is_superhost'].unique()
print(unique_values)
data_types = df['room_type'].apply(type).unique()
print("Tipovi podataka u koloni:", data_types) 
data_types = df['room_private'].apply(type).unique()
print("Tipovi podataka u koloni:", data_types)'''

print("\n")
################################################





