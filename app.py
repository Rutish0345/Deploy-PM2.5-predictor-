from flask import Flask, request, render_template
import os, joblib, numpy as np

app = Flask(__name__)

# Lazy loading: carga el modelo solo al primer request (ahorra RAM en startup)
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load('modelo/pipeline_pm25.pkl')
    return _pipeline

FEATURES = ['DEWP', 'month', 'day', 'TEMP', 'Iws', 'PRES']

FIELDS = [
    {'name': 'DEWP',  'label': 'Punto de rocio',            'unit': 'C',    'min': -40,  'max': 28,    'step': '0.1', 'example': '2'},
    {'name': 'month', 'label': 'Mes',                        'unit': '1-12', 'min': 1,    'max': 12,    'step': '1',   'example': '1'},
    {'name': 'day',   'label': 'Dia del mes',                'unit': '1-31', 'min': 1,    'max': 31,    'step': '1',   'example': '15'},
    {'name': 'TEMP',  'label': 'Temperatura',                'unit': 'C',    'min': -19,  'max': 42,    'step': '0.1', 'example': '-5'},
    {'name': 'Iws',   'label': 'Velocidad del viento (Iws)', 'unit': 'm/s',  'min': 0.45, 'max': 565.5, 'step': '0.01', 'example': '5.37'},
    {'name': 'PRES',  'label': 'Presion atmosferica',        'unit': 'hPa',  'min': 991,  'max': 1046,  'step': '0.1', 'example': '1016'},
]


def nivel_calidad(pm25):
    if pm25 <= 12:
        return 'Buena', '#2ecc71'
    elif pm25 <= 35.4:
        return 'Moderada', '#f1c40f'
    elif pm25 <= 55.4:
        return 'No saludable para grupos sensibles', '#e67e22'
    elif pm25 <= 150.4:
        return 'No saludable', '#e74c3c'
    elif pm25 <= 250.4:
        return 'Muy no saludable', '#8e44ad'
    else:
        return 'Peligrosa', '#922b21'


@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    nivel = None
    color = None
    error = None
    valores = {}

    if request.method == 'POST':
        try:
            for f in FIELDS:
                val = request.form.get(f['name'], '').strip()
                if val == '':
                    raise ValueError(f"El campo '{f['label']}' es obligatorio.")
                valores[f['name']] = float(val)

            if not (1 <= valores['month'] <= 12):
                raise ValueError('El mes debe estar entre 1 y 12.')
            if not (1 <= valores['day'] <= 31):
                raise ValueError('El dia debe estar entre 1 y 31.')
            if valores['Iws'] < 0:
                raise ValueError('La velocidad del viento no puede ser negativa.')

            # Prediccion usando numpy (sin pandas para ahorrar RAM)
            X = np.array([[valores[f] for f in FEATURES]])
            pred = get_pipeline().predict(X)[0]
            prediccion = max(0, round(float(pred), 2))
            nivel, color = nivel_calidad(prediccion)

        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = f'Error interno: {str(e)}'

    return render_template('index.html',
                           fields=FIELDS,
                           prediccion=prediccion,
                           nivel=nivel,
                           color=color,
                           error=error,
                           valores=valores)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
