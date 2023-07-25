from flask import Flask, Blueprint, render_template, request
from flask_login import login_required, current_user
from .models import TECParams
from . import db
import pickle, random, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
import atexit

month_to_day = {"1":0, "2":31, "3":59, "4":90, "5":120, "6":151, "7":181, "8":212, "9":243, "10":273, "11":304, "12":334}
views = Blueprint('views', __name__)

#creating an instance of flask app
app = Flask(__name__)

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    return render_template("home.html", user=current_user)

#defining the range of input parameters:
year_range = range(2001, 2031)
month_range = range(1, 366)
day_range = range(1, 366)
hour_range = range(0, 24)
rz_12_range = range(0, 401)
ig_12_range = range(0, 401)
ap_index_range = range(0, 101)
kp_index_range = range(0, 101)

@views.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    if request.method == 'POST':
        year = request.form['year']
        day_of_year = month_to_day[request.form['month']] + int(request.form['day'])
        hour_of_day = int(request.form['hour_of_day'])
        rz_12 = request.form['rz_12']
        ig_12 = request.form['ig_12']
        ap_index = request.form['ap_index']
        kp_index = request.form['kp_index']
        with open("website\model\TEC_model4.pkl", "rb") as file:
            current_model = pickle.load(file)
        inputs = np.array([[year, day_of_year, hour_of_day, rz_12, ig_12, ap_index, kp_index]])
        inputs = inputs.reshape(1, -1)
        prediction = current_model.predict(inputs) # Passing in variables for prediction
        tec_output = prediction
        tec_input = TECParams(year=year,
                              day_of_year=day_of_year,
                              hour_of_day=hour_of_day,
                              rz_12=rz_12,
                              ig_12=ig_12,
                              ap_index=ap_index,
                              kp_index=kp_index,
                              tec_output=tec_output
                                     )
        db.session.add(tec_input)
        db.session.commit()
    return render_template('home.html', prediction_text='the value of TEC content is {}'.format(tec_output), user = current_user)

@views.route("/about")
@login_required
def about():
    return render_template("about.html", user=current_user)

@views.route("/plot", methods = ['GET', 'POST'])
@login_required
def plot():
    if request.method == 'POST':
            tec_data = pd.read_csv('website\\model\\data_to_kaggle.csv')

            X = tec_data['Y']  #contains TEC values for all observations
            features = ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 'X_20', 'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27', 'X_28', 'X_29', 'X_30', 'X_31', 'X_32', 'X_33', 'X_34', 'X_35', 'X_36', 'X_37', 'X_38', 'X_39', 'X_40', 'X_41', 'X_42', 'X_43', 'X_44', 'X_45', 'X_46', 'X_47', 'X_48', 'X_49', 'X_50', 'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56', 'X_57', 'X_58', 'X_59', 'X_60', 'X_61', 'X_62', 'X_63', 'X_64', 'X_65', 'X_66', 'X_67', 'X_68', 'X_69', 'X_70', 'X_71', 'X_72', 'X_73', 'X_74', 'X_75', 'X_76', 'X_77', 'X_78', 'X_79', 'X_80', 'X_81', 'X_82', 'X_83', 'X_84', 'X_85', 'X_86', 'X_87', 'X_88', 'X_89', 'X_90', 'X_91', 'X_92', 'X_93', 'X_94', 'X_95', 'X_96', 'X_97', 'X_98', 'X_99', 'X_100', 'X_101', 'X_102', 'X_103', 'X_104', 'X_105', 'X_106', 'X_107', 'X_108', 'X_109', 'X_110', 'X_111', 'X_112', 'X_113', 'X_114', 'X_115', 'X_116', 'X_117', 'X_118', 'X_119', 'X_120', 'X_121', 'X_122', 'X_123', 'X_124', 'X_125', 'X_126', 'X_127', 'X_128', 'X_129', 'X_130', 'X_131', 'X_132', 'X_133', 'X_134', 'X_135', 'X_136', 'X_137', 'X_138', 'X_139', 'X_140', 'X_141', 'X_142', 'X_143', 'X_144', 'X_145', 'X_146', 'X_147', 'X_148', 'X_149', 'X_150', 'X_151', 'X_152', 'X_153', 'X_154', 'X_155', 'X_156', 'X_157', 'X_158', 'X_159', 'X_160', 'X_161', 'X_162', 'X_163', 'X_164', 'X_165', 'X_166', 'X_167', 'X_168', 'X_169', 'X_170', 'X_171', 'X_172', 'X_173', 'X_174', 'X_175', 'X_176', 'X_177', 'X_178', 'X_179', 'X_180', 'X_181', 'X_182', 'X_183', 'X_184', 'X_185', 'X_186', 'X_187', 'X_188', 'X_189', 'X_190']
            y = tec_data[features] #contains values of all features

            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.33, random_state=1) # 3 parts go to testing and 7 parts go to training for every 10 parts

            # Opening saved model
            with open("website\\model\\TEC_model2.pkl", "rb") as file:
                current_model = pickle.load(file)

            prediction = current_model.predict(y_test) # Passing in variables for prediction
            inputval = random.randint(0, 401)
            print("The result is",prediction[inputval]) # Printing result
            def predictor():
                inputval = random.randint(0, 401)
                return prediction[inputval]
            output2 = pd.Series([], dtype='object')
            abc = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            for i in abc:
                output2[i] = predictor()
            print(output2)
            fig, ax = plt.subplots()
            output2.plot.line()
            plt.xlabel("Days -->")
            plt.ylabel("TEC value (10^6e-)")
            tmpfile = BytesIO()
            fig.savefig(tmpfile, format='png')
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            prediction_output =  '{% extends "base.html" %} {% block title %}Home{% endblock %} {% block content%}'+ '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '</body>{% endblock %}'

            with open('test.html','w') as f:
                f.write(prediction_output)
            shutil.move(r"test.html", r"website\\templates\\test.html")
            return render_template('test.html', user = current_user)

    return render_template("plot.html", user = current_user)

def close_figure():
    plt.close('all')

atexit.register(close_figure)