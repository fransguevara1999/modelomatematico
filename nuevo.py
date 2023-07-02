from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
import json
# Crear la aplicación Flask
app = Flask(__name__)

# Cargar los datos desde el archivo CSV
data = pd.read_csv('modeloMatematico3.csv', delimiter=';')

# Convertir la columna 'op' a cadena de caracteres (string)
data['resultado'] = data['resultado'].astype(str)
# Reemplazar la coma por un punto en la columna 'op'
data['resultado'] = data['resultado'].str.replace(',', '.').astype(float)

data['op'] = data['op'].astype(int)

# Extraer las variables de entrada (X) y los resultados (y)
dataset = data.values
x = dataset[:, [0, 1, 5]].astype(float)
y = dataset[:, [5, 6]].astype(float)

# Normalizar las características
x_normalized = (x - x.mean()) / x.std()

x_normalized = x_normalized.astype(float)

# Construir el modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])

# Compilar el modelo
model.compile(optimizer='adam', loss=['mse', 'mse'])

# Entrenar el modelo
model.fit(x_normalized, y, epochs=100)

# Ruta de la API para obtener op con resultados similares


@app.route('/get_similar_ops', methods=['POST'])
def get_similar_ops():
    data = request.json
    variable1 = data['variable1']
    variable2 = data['variable2']
    op = data['op']
    input_data = pd.DataFrame(
        {'variable1': [variable1], 'variable2': [variable2], 'op': [op]})
    input_data_normalized = (input_data - x.mean()) / x.std()
    input_data_normalized = input_data_normalized.astype(float)
    predicted_results = model.predict(input_data_normalized)

    response = []
    for result in predicted_results:
        response.append({'op': int(result[0]), 'resultado': int(result[1])})

    return jsonify(response)


# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run()
