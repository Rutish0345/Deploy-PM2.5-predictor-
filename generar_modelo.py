import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import joblib, os

SEED     = 42
URL      = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv'
FEATURES = ['DEWP', 'month', 'day', 'TEMP', 'Iws', 'PRES']

print("Descargando dataset Beijing PM2.5...")
df = pd.read_csv(URL)

df_m  = df.drop(columns=['No'])
y_raw = df_m['pm2.5']
X_raw = df_m.drop(columns=['pm2.5'])
mask  = y_raw.notnull()
X_cl  = X_raw[mask].reset_index(drop=True)
y_cl  = y_raw[mask].reset_index(drop=True)
X_fin = X_cl[FEATURES]

print(f"Entrenando modelo con {len(X_fin):,} registros...")
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=SEED))
])
pipeline.fit(X_fin, y_cl)

os.makedirs('modelo', exist_ok=True)
out = 'modelo/pipeline_pm25.pkl'
joblib.dump(pipeline, out, compress=3)
print(f"Modelo guardado en: {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

ej = pd.DataFrame([{'DEWP': 2.0, 'month': 1, 'day': 15,
                     'TEMP': -5.0, 'Iws': 5.37, 'PRES': 1016.0}])
print(f"Prediccion de prueba: PM2.5 = {pipeline.predict(ej)[0]:.2f} ug/m3")
print("Listo.")