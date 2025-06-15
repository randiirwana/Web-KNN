import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Buat folder static jika belum ada
if not os.path.exists('static'):
    os.makedirs('static')

# Set style untuk plot
plt.style.use('default')
sns.set_theme()

# Load dan proses data
df = pd.read_csv('Dataset.csv')
df = df.drop(columns=['Loan_ID'])

# Visualisasi distribusi target sebelum encoding
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Loan_Status')
plt.title('Distribusi Status Pinjaman')
plt.xlabel('Status Pinjaman')
plt.ylabel('Jumlah')
plt.savefig('static/distribusi_target.png')
plt.close()

# Visualisasi korelasi antar fitur numerik
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Korelasi Antar Fitur Numerik')
plt.tight_layout()
plt.savefig('static/korelasi_fitur.png')
plt.close()

# Perbaikan penanganan missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df.loc[:, col] = df[col].fillna(df[col].mode().iloc[0])
    else:
        df.loc[:, col] = df[col].fillna(df[col].mean())

# Encode kolom kategorikal termasuk target
le = LabelEncoder()
categorical_columns = df.select_dtypes(include='object').columns
for col in categorical_columns:
    df.loc[:, col] = le.fit_transform(df[col])

# Pisahkan fitur dan target
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# Pastikan target adalah integer
y = y.astype(int)

# Print informasi data
print("\nInformasi Data:")
print("Jumlah sampel:", len(df))
print("Jumlah fitur:", X.shape[1])
print("Kelas target:", np.unique(y))
print("Distribusi kelas:", np.bincount(y))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Parameter tuning dengan GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Visualisasi hasil parameter tuning
cv_results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(12, 6))
sns.boxplot(data=cv_results, x='param_n_neighbors', y='mean_test_score')
plt.title('Akurasi Model untuk Berbagai Nilai K')
plt.xlabel('Jumlah Tetangga (K)')
plt.ylabel('Akurasi')
plt.savefig('static/parameter_tuning.png')
plt.close()

# Print hasil parameter terbaik
print("\nHasil Parameter Tuning:")
print("Parameter terbaik:", grid_search.best_params_)
print("Akurasi terbaik pada validasi:", grid_search.best_score_)

# Gunakan model dengan parameter terbaik
best_model = grid_search.best_estimator_

# Prediksi dan evaluasi
y_pred = best_model.predict(X_test)

# Visualisasi confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.savefig('static/confusion_matrix.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Simpan model dan scaler
joblib.dump(best_model, 'model/knn_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(grid_search.best_params_, 'model/best_params.pkl')
joblib.dump(le, 'model/label_encoder.pkl')

print("\nGrafik telah disimpan di folder 'static':")
print("1. distribusi_target.png - Distribusi status pinjaman")
print("2. korelasi_fitur.png - Korelasi antar fitur numerik")
print("3. parameter_tuning.png - Hasil tuning parameter K")
print("4. confusion_matrix.png - Confusion matrix model")
