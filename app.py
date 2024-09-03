import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle as pkl



df = pd.read_csv("synthetic_data.csv")
print(df.head())
encoder = LabelEncoder()
df_recommened=encoder.fit_transform(df.recommended_subject)
df_recommened = pd.DataFrame(df_recommened, columns=['new_recommended_subject'])
df = pd.concat([df, df_recommened], axis=1)


X = df[["java_score","dotnet_score","data_engineering_score"]]
y = df['new_recommended_subject']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

with open("model.pkl",'wb')as f:
    pkl.dump(model,f)
with open("encoder.pkl",'wb') as f:
    pkl.dump(encoder,f)