import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv('Dataset.csv')  # File dari Kaggle
df.drop(columns=['Loan_ID'], inplace=True)

# 2. Tangani missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# 3. Encode kolom kategorikal
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# 4. Pisah fitur dan target
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# 5. Normalisasi fitur numerik
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# 6. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Train model KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# 8. Evaluasi model
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
