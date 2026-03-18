import pandas as pd
import numpy as np

# Mock data for BostonHousing.csv
bh = pd.DataFrame(np.random.rand(100, 13), columns=['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat'])
bh['medv'] = np.random.rand(100) * 50
bh.to_csv('BostonHousing.csv', index=False)
print("Created BostonHousing.csv")

# Mock data for Mall_Customers.csv
mc = pd.DataFrame({
    'CustomerID': range(1, 201),
    'Gender': np.random.choice(['Male', 'Female'], 200),
    'Age': np.random.randint(18, 70, 200),
    'Annual Income (k$)': np.random.randint(15, 140, 200),
    'Spending Score (1-100)': np.random.randint(1, 100, 200)
})
mc.to_csv('Mall_Customers.csv', index=False)
print("Created Mall_Customers.csv")
