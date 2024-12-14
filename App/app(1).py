# Импортируем необходимые библиотеки
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template

# Создаем Flask приложение
app = Flask(__name__)

# Глобальная переменная для хранения загруженной модели
model = None

def load_model():
    """Загрузка модели"""
    global model
    model_path = "DS/App/saved_models/keras_model"
    try:
        model = tf.keras.models.load_model(model_path)
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        model = None

def predict(params):
    """Функция для предсказания"""
    try:
        # Проверяем, что модель загружена
        if model is None:
            raise ValueError("Модель не загружена!")
        
        # Прогноз на основе входных данных
        prediction = model.predict([params])
        return prediction[0][0]
    except Exception as e:
        return f"Ошибка предсказания: {str(e)}"

@app.route('/', methods=['POST', 'GET'])
def app_calculation():
    param_lst = []
    message = ''
    if request.method == 'POST':
        try:
            # Получаем данные из формы
            for i in range(1, 13):
                param = request.form.get(f'param{i}')
                param_lst.append(float(param))
            
            # Вызываем функцию предсказания
            message = predict(param_lst)
        except ValueError:
            message = "Проверьте введенные данные. Все параметры должны быть числами."
        except Exception as e:
            message = f"Произошла ошибка: {str(e)}"
    
    # Отображаем HTML шаблон
    return render_template("index.html", message=message)

# Запускаем приложение
if __name__ == "__main__":
    load_model()  # Загружаем модель перед запуском приложения
    app.run(debug=True)
