import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


df = pd.read_csv(r"E:\third\car.csv")

df['Years'] = datetime.now().year - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)

fuel_encoder = LabelEncoder()
seller_encoder = LabelEncoder()
trans_encoder = LabelEncoder()

df['Fuel_Type'] = fuel_encoder.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = seller_encoder.fit_transform(df['Seller_Type'])
df['Transmission'] = trans_encoder.fit_transform(df['Transmission'])


X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor()
model.fit(X_train, y_train)

new_raw = pd.DataFrame([
    {
        'Present_Price': 5.59,
        'Kms_Driven': 27000,
        'Owner': 0,
        'Fuel_Type': 'Petrol',
        'Seller_Type': 'Individual',
        'Transmission': 'Manual',
        'Years': 7
    },
    {
        'Present_Price': 7.1,
        'Kms_Driven': 43000,
        'Owner': 1,
        'Fuel_Type': 'Diesel',
        'Seller_Type': 'Dealer',
        'Transmission': 'Automatic',
        'Years': 5
    }
])


new_raw['Fuel_Type'] = fuel_encoder.transform(new_raw['Fuel_Type'])
new_raw['Seller_Type'] = seller_encoder.transform(new_raw['Seller_Type'])
new_raw['Transmission'] = trans_encoder.transform(new_raw['Transmission'])

new_raw = new_raw[X.columns]

predictions = model.predict(new_raw)

output = new_raw.copy()
output['Predicted_Selling_Price'] = predictions.round(2)

output['Fuel_Type'] = fuel_encoder.inverse_transform(output['Fuel_Type'])
output['Seller_Type'] = seller_encoder.inverse_transform(output['Seller_Type'])
output['Transmission'] = trans_encoder.inverse_transform(output['Transmission'])

print("\n--- Predicted Selling Prices ---\n")
print(output[['Fuel_Type', 'Years', 'Transmission', 'Seller_Type', 'Kms_Driven', 'Owner', 'Present_Price', 'Predicted_Selling_Price']])

sample = X_test.iloc[[0]]
actual_price = y_test.iloc[0]
predicted_price = model.predict(sample)[0]

sample_display = sample.copy()
sample_display['Fuel_Type'] = fuel_encoder.inverse_transform([sample_display['Fuel_Type'].values[0]])[0]
sample_display['Seller_Type'] = seller_encoder.inverse_transform([sample_display['Seller_Type'].values[0]])[0]
sample_display['Transmission'] = trans_encoder.inverse_transform([sample_display['Transmission'].values[0]])[0]

print("\n--- Sample Prediction vs Actual ---\n")
print(sample_display)
print(f"\nPredicted Price: {round(predicted_price, 2)}")
print(f"Actual Price:    {round(actual_price, 2)}")
