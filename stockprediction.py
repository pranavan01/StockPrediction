# import all required libraries
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Take user input for which stock they wish they invest in
stock=input("What stock do u want advise on:")

# Get stock data using Quandl
df = quandl.get("WIKI/"+stock)

# A variable for predicting 'n' days out into the future
forcast_out = 30

# Create a new column that is shifted up by forecast_out to create the "Predicted" stock prices
df['Prediction']=df[['Adj. Close']].shift(-forcast_out)

# Create the independent data set (X) by considering the Adj. Close prices
# Convert the Dataframe stock values into a 2 dimensional NumPy array
X = np.array(df.drop(['Prediction'],1))
# Remove the last  "n" days from the array
X = X[:-forcast_out]

# Create the dependent data set (Y) by considering the Prediction column
# Convert the Dataframe stock values into NumPy array which is 1 dimensional
y= np.array(df['Prediction'])
# Remove the last "n" days from the array
y=y[:-forcast_out]

# Split the data into 20% Testing and 80% Training
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# Create and train the Linear Regression Model
lr=LinearRegression()
# Train the model
lr.fit(x_train,y_train)

# Get the last n days from the Adj Close prices and convert it into a 2 dimensional NumPy Array
x_forcast=np.array(df.drop(['Prediction'],1))[-forcast_out:]

# Use the linear regression model to predict the next n days
lr_prediciton=lr.predict(x_forcast)

# Determine whether or not user should invest in the specific stock
sum_price=0
for x in range (len(lr_prediciton)):
    sum+=lr_prediciton[x]
avg_val=sum_price/len(lr_prediciton)
if avg_val >= lr_prediciton[0]:
    print("Invest in this Stock")
else:
    print("Don't Invest in this stock")

# Plot a graph of the stock prices for the next 30 days using Matplotlib
plt.plot(lr_prediciton)
plt.xlabel('Days from Now')
plt.ylabel('Price per Stock ($)')
plt.show()