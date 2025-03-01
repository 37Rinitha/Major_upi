import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from flask import Flask, redirect, request, render_template, session, flash
from flask_login import LoginManager, login_required, login_user, logout_user, current_user, UserMixin
from flask_mysqldb import MySQL

dataset = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)

x = dataset.iloc[:, :10].values
y = dataset.iloc[:, 10].values

scaler = StandardScaler()
scaler.fit_transform(x)

model = tf.keras.models.load_model('src/model/project_model1.keras')

app = Flask(__name__)
app.secret_key = 'super secret string'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'fraud_detection'
app.config['MYSQL_HOST'] = 'localhost'
mysql = MySQL(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class Users(UserMixin):
    def __init__(self, user_id, password, name):
        self.user_id = user_id
        self.password = password
        self.name = name

    @staticmethod
    def get(user_id):
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
        user = cur.fetchone()
        if user:
            return Users(user_id=user[0], password=user[1], name=user[2])
        return None

    def get_id(self):
        return self.user_id

@login_manager.user_loader
def load_user(user_id):
    return Users.get(user_id)

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['uname']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE user_id = %s AND password = %s", (user_id, password))
        user = cur.fetchone()
        if user:
            user_obj = Users(user_id=user[0], password=user[1], name=user[2])
            login_user(user_obj)
            session['user_id'] = user_id
            return redirect('/upload')
        else:
            return render_template('login.html', messages='Invalid credentials')
    return render_template('login.html')

@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

@app.route('/validate', methods=["POST"])
def validate():
    if request.method == 'POST':
        user_id = request.form['uname']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE user_id = %s AND password = %s", (user_id, password))
        user = cur.fetchone()
        if user:
            user_obj = Users(user_id=user[0], password=user[1], name=user[2])
            login_user(user_obj)
            session['user_id'] = user_id
            return redirect('/upload')
        else:
            return render_template('login.html',message='Invalid credentials')

@app.route('/preview', methods=["POST"])
@login_required
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df)

@app.route('/prediction1', methods=['GET'])
@login_required
def prediction1():
    return render_template('index.html')

@app.route('/chart')
@login_required
def chart():
    return render_template('chart.html')

@app.route('/detect', methods=['POST'])
@login_required
def detect():
    trans_datetime = pd.to_datetime(request.form.get("trans_datetime"))
    v1 = trans_datetime.hour
    v2 = trans_datetime.day
    v3 = trans_datetime.month
    v4 = trans_datetime.year
    v5 = int(request.form.get("category"))
    v6 = float(request.form.get("card_number"))
    dob = pd.to_datetime(request.form.get("dob"))
    v7 = np.round((trans_datetime - dob) / np.timedelta64(1, 'D') / 365.25)
    v8 = float(request.form.get("trans_amount"))
    v9 = int(request.form.get("state"))
    v10 = int(request.form.get("zip"))
    x_test = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])
    y_pred = model.predict(scaler.transform([x_test]))
    if y_pred[0][0] <= 0.5:
        result = "VALID TRANSACTION"
    else:
        result = "FRAUD TRANSACTION"
    return render_template('result.html', OUTPUT='{}'.format(result))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect('/login')

if __name__ == "__main__":
    app.run(debug=True)
