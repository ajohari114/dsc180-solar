#List of Imports
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from Colormetrics import process_pictures
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

def mse_2d(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    return ((true-pred)**2).mean()

c_metrics = process_pictures('gopro_images')

#Reading in features from the PASCAL data
f = pd.read_json('maestro_sample_log.json')
df = f.transpose()
def extract_details(row):
    return row['worklist'][1]['details']['drops']
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
df = df.apply(extract_details,axis=1)
flattened_series = df.apply(lambda x: flatten_dict(x[0]))

# Sort the DataFrame index numerically
df_sorted = flattened_series.reindex(sorted(flattened_series.index, key=lambda x: int(x[6:])))

# Flatten the dictionary of composition data into their own columns
flattened_df = pd.DataFrame(df_sorted.tolist(), index=df_sorted.index)

#Drop specific samples based on colormetrics file
samples_to_ignore = ['sample17','sample22','sample38','sample39','sample40','sample44','sample41','sample42','sample43']
samples_to_keep = ~flattened_df.index.isin(samples_to_ignore)
cleaned = flattened_df[samples_to_keep]

new_col = [c_metrics.loc[:,i].values for i in cleaned.index]
cleaned['colormetrics'] = new_col

# Defining features(X) and target variable(y)
X = cleaned[['solution_solutes', 'solution_molarity', 'solution_well_well','solution_well_labware']]
y = cleaned['colormetrics'].values

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = [list(i) for i in y_train]
y_test = [list(i) for i in y_test]


# Preprocessing the data based on data type
preprocessor = ColumnTransformer(
    transformers=[
        ('solutes', OneHotEncoder(), ['solution_solutes', 'solution_well_well','solution_well_labware']),
        ('numerical', StandardScaler(), ['solution_molarity'])
    ],
)
#list of types of regressors to try
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet': ElasticNet(),
    'Huber Regressor': HuberRegressor(max_iter=200),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(),
    'SVR': SVR(),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
}

best_model = None
best_mse = np.inf

test_errors = []

for name, regressor in regressors.items():
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', MultiOutputRegressor(regressor))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mse_2d(y_test, y_pred)
    print(f"{name} - MSE on the test set: {mse}")
    
    test_errors.append(mse)
    
    if mse < best_mse:
        best_model = model
        best_mse = mse
        
        
temp_df = pd.DataFrame()
temp_df['regressor'] = regressors.keys()
temp_df['test_error'] = test_errors
temp_df = temp_df.sort_values('test_error', ascending = False)
plt.barh(temp_df['regressor'], temp_df['test_error'], log = True)
plt.title('Test Error of Each Regressor (Lower is Better)')
plt.xlabel('Mean Squared Error')
