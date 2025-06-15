from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import joblib
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('model/knn_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    data_dict = {
        'Jenis Kelamin': 'Laki-laki' if int(request.form['Gender']) == 1 else 'Perempuan',
        'Status Menikah': 'Ya' if int(request.form['Married']) == 1 else 'Tidak',
        'Jumlah Tanggungan': request.form['Dependents'],
        'Pendidikan': 'Tidak Lulus' if int(request.form['Education']) == 1 else 'Lulus',
        'Wiraswasta': 'Ya' if int(request.form['Self_Employed']) == 1 else 'Tidak',
        'Pendapatan Pemohon': request.form['ApplicantIncome'],
        'Pendapatan Penjamin': request.form['CoapplicantIncome'],
        'Jumlah Pinjaman': request.form['LoanAmount'],
        'Jangka Waktu Pinjaman': request.form['Loan_Amount_Term'],
        'Riwayat Kredit': 'Baik' if float(request.form['Credit_History']) == 1 else 'Buruk',
        'Area Properti': 'Perkotaan' if int(request.form['Property_Area']) == 0 else ('Pinggir Kota' if int(request.form['Property_Area']) == 1 else 'Desa')
    }
    data = [
        int(request.form['Gender']),
        int(request.form['Married']),
        int(request.form['Dependents']),
        int(request.form['Education']),
        int(request.form['Self_Employed']),
        float(request.form['ApplicantIncome']),
        float(request.form['CoapplicantIncome']),
        float(request.form['LoanAmount']),
        float(request.form['Loan_Amount_Term']),
        float(request.form['Credit_History']),
        int(request.form['Property_Area'])
    ]
    input_array = np.array([data])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    result = "LAYAK" if prediction[0] == 1 else "TIDAK LAYAK"
    return render_template('result.html', result=result, data=data_dict)

@app.route('/grafik')
def grafik():
    # --- Distribusi Target ---
    df = pd.read_csv('Dataset.csv')
    distribusi = df['Loan_Status'].value_counts()
    bar = go.Figure([go.Bar(x=distribusi.index, y=distribusi.values)])
    bar.update_layout(title='Distribusi Status Pinjaman', xaxis_title='Status', yaxis_title='Jumlah')
    distribusi_html = pio.to_html(bar, full_html=False)

    # --- Korelasi Fitur ---
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlation = df[numeric_cols].corr()
    corr_fig = go.Figure(data=go.Heatmap(
        z=correlation.values,
        x=correlation.columns,
        y=correlation.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=correlation.round(2).values,
        texttemplate='%{text}',
        textfont={"size":10}
    ))
    corr_fig.update_layout(title='Korelasi Antar Fitur Numerik', height=800, width=1000)
    corr_html = pio.to_html(corr_fig, full_html=False)

    # --- Parameter Tuning ---
    le = joblib.load('model/label_encoder.pkl')
    df = df.drop(columns=['Loan_ID'])
    for col in df.columns:
        if df[col].dtype == 'object':
            df.loc[:, col] = df[col].fillna(df[col].mode().iloc[0])
        else:
            df.loc[:, col] = df[col].fillna(df[col].mean())
    categorical_columns = df.select_dtypes(include='object').columns
    for col in categorical_columns:
        df.loc[:, col] = le.fit_transform(df[col])
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status'].astype(int)
    X_scaled = scaler.transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0', 'Pred 1'], y=['Actual 0', 'Actual 1'], colorscale='Blues', showscale=True))
    cm_fig.update_layout(title='Confusion Matrix', xaxis_title='Prediksi', yaxis_title='Aktual')
    cm_html = pio.to_html(cm_fig, full_html=False)

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    tuning_fig = go.Figure()
    for metric in ['euclidean', 'manhattan', 'minkowski']:
        for weight in ['uniform', 'distance']:
            mask = (cv_results['param_metric'] == metric) & (cv_results['param_weights'] == weight)
            tuning_fig.add_trace(go.Scatter(
                x=cv_results[mask]['param_n_neighbors'],
                y=cv_results[mask]['mean_test_score'],
                name=f'{metric}-{weight}',
                mode='lines+markers'
            ))
    tuning_fig.update_layout(
        title='Akurasi Model untuk Berbagai Parameter',
        xaxis_title='Jumlah Tetangga (K)',
        yaxis_title='Akurasi',
        height=600
    )
    tuning_html = pio.to_html(tuning_fig, full_html=False)

    return render_template(
        'grafik.html',
        distribusi_html=distribusi_html,
        corr_html=corr_html,
        tuning_html=tuning_html,
        cm_html=cm_html
    )

if __name__ == "__main__":
    app.run(debug=True)
