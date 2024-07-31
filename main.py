# %%
import pickle

with open('./mod_05_topic_10_various_data.pkl', 'rb') as fl:
    data = pickle.load(fl)
    
autos = data['autos']

# %%

X = autos.copy()
y = X.pop('price')

# %%

cat_features = X.select_dtypes(include='object').columns

for col in cat_features:
    X[col], __doc__ = X[col].factorize()
    
# %%
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

mi = mutual_info_regression(X, y, 
                            discrete_features=X.columns.isin(
                                cat_features.to_list() +
                                ['num_of_doors',
                                 'num_of_cylinders']),
                            random_state=42)
    
mi = pd.Series(mi, name='MI scores', index=X.columns).sort_values()

mi.sample(5)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42)

# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error

model = GradientBoostingRegressor(learning_rate=0.3,
                                  subsample=0.75,
                                  max_features='sqrt',
                                  random_state=42).fit(X_train, y_train)

y_pred = model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, y_pred)
print(mape)

# %%
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()

comparison = pd.DataFrame({'MI scores': mi, 'Feature Importances': feature_importances})

comparison['default_rank'] = comparison['Feature Importances'].rank()
comparison['max_rank'] = comparison['Feature Importances'].rank(method='max')
comparison['NA_bottom'] = comparison['Feature Importances'].rank(na_option='bottom')
comparison['pct_rank'] = comparison['Feature Importances'].rank(pct=True)

print(comparison)

# %%
import seaborn as sns
import matplotlib.pyplot as plt 

comparison_long = pd.melt(comparison.reset_index(), id_vars='index', value_vars=['MI scores', 'Feature Importances'], 
                          var_name='Metric', value_name='Value')

sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=comparison_long, kind="bar",
    x="index", y="Value", hue="Metric",
    palette="dark", alpha=.6, height=6,
    aspect=2
)
g.despine(left=True)
g.set_xticklabels(rotation=90)
g.set_axis_labels("Features", "Scores")
g.legend.set_title("Metrics")
plt.title('Comparison of MI Scores and Feature Importances')
plt.show()
