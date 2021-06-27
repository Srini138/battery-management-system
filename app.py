from flask import Flask,render_template,request
import pickle as pk
import numpy as np

app=Flask(__name__)

model1=pk.load(open('capacity_prediction.pkl','rb'))
model2=pk.load(open('cycle_prediction.pkl','rb'))
capacity_model=pk.load(open('capacity_prediction_min_max_scaler.pkl','rb'))
cycle_model=pk.load(open('cycle_prediction_min_max_scaler.pkl','rb'))

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/description')
def description():
    return render_template('description.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result',methods=['POST'])
def result():
    charge_time=float(request.form['cmax_time'])
    discharge_time=float(request.form['dmax_time'])
    charge_voltage_time=float(request.form['cmax_vtime'])
    discharge_voltage_time=float(request.form['dmax_vtime'])

    capacity_pred=(charge_time,discharge_time,charge_voltage_time,discharge_voltage_time)
    capacity_pred_numpy=np.asarray(capacity_pred)
    capacity_pred_reshape=capacity_pred_numpy.reshape(1,-1)
    std_data=capacity_model.transform(capacity_pred_reshape)

    predict=model1.predict(std_data)
    predictions=round(predict[0],4)


    cycle_pred=(charge_time,discharge_time,charge_voltage_time,discharge_voltage_time,predictions)
    cycle_pred_numpy=np.asarray(cycle_pred)
    cycle_pred_reshape=cycle_pred_numpy.reshape(1,-1)
    std_data1=cycle_model.transform(cycle_pred_reshape)

    predict1=model2.predict(std_data1)
    predictions1=round(predict1[0])

    return render_template('result.html',data=[predictions,predictions1])


if __name__ == '__main__':
    app.run(debug=True)
