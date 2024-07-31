# %%
import pickle

with open('./mod_05_topic_10_various_data.pkl', 'rb') as fl:
    data = pickle.load(fl)
    
autos = data['autos']

# %%
autos['stroke_ratio'] = autos['stroke'] / autos['bore']
autos['make_and_style'] = autos['make'] + '_' + autos['body_style']
   
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

X_train, X_test, y_train, y_test = train_test_split(autos.drop('price', axis=1), y, 
                                                    test_size=0.33, 
                                                    random_state=42)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform='pandas')

X_train_num = X_train.select_dtypes(exclude='object')
X_test_num = X_test.select_dtypes(exclude='object')

X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# %%
from category_encoders import TargetEncoder
encoder = TargetEncoder()

X_train_cat = X_train.select_dtypes(include='object')
X_test_cat = X_test.select_dtypes(include='object')

X_train_cat = encoder.fit_transform(X_train_cat, y_train)
X_test_cat = encoder.transform(X_test_cat)

# %%
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(learning_rate=0.3,
                                  subsample=0.75,
                                  max_features='sqrt',
                                  random_state=42).fit(
                                      pd.concat([X_train_cat, X_train_num], axis=1),
                                      y_train)

# %%
from sklearn.ensemble import RandomForestRegressor

model_forest = RandomForestRegressor(random_state=42)
model_forest.fit( pd.concat([X_train_cat, X_train_num], axis=1), y_train)

# Feature importances
importances = pd.Series(model_forest.feature_importances_, index=X.columns).sort_values(ascending=False)


# %%
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6, 8))
plt.barh(np.arange(len(mi)), mi)
plt.yticks(np.arange(len(mi)), mi.index)
plt.title('Mutual Information Scores')

plt.show()

# %%
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()

comparison = pd.DataFrame({'MI scores': mi, 'Feature Importances': feature_importances})

comparison['pct_rank'] = comparison['Feature Importances'].rank(pct=True)

print(comparison)

# %%
import seaborn as sns
import matplotlib.pyplot as plt 

comparison_long = pd.melt(comparison.reset_index(), id_vars='index', value_vars=['MI scores', 'Feature Importances'], 
                          var_name='Metric', value_name='Value')

sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=comparison_long, kind="bar", orient='y',
    x="Value", y="index", hue="Metric",
    palette="dark", alpha=.6, height=6,
    aspect=2
)
g.despine(left=True)
g.set_xticklabels(rotation=90)
g.set_axis_labels("Features", "Scores")
g.legend.set_title("Metrics")
plt.title('Comparison of MI Scores and Feature Importances')
plt.show()
