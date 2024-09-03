import pickle as pkl
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Enter the details")
print("============================================")

input_marks = input("Enter the marks in Java, .NET, Data Engineeering: ")
marks = [int(i) for i in input_marks.split(' ')]

marks = np.array([marks])

with open("model.pkl", "rb") as f:
    model = pkl.load(f)

y_pred = model.predict(marks)

with open("encoder.pkl", 'rb') as f:
    encoder = pkl.load(f)

decoder = encoder.inverse_transform(y_pred)

print("---------------------------------------------")
print(f"Subject need to be focused on : {decoder}")