
from django.shortcuts import render
from joblib import load

model = load('C:/Users/Admin/deployChap12/savedModel/model12.joblib', mmap_mode='r')
# Create your views here.
def predict(request):
    return render(request, 'predict.html')
def result(request):
    Glucose= request.GET['glucose']
    BMI = request.GET['bmi']
    Age = request.GET['age']
    y_pred = model.predict([[Glucose, BMI, Age]])
    print('ok')
    return render(request, 'result.html', {'predict': y_pred})