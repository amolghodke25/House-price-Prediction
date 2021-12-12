# House-price-Prediction
Using Linear regression we discuss the various Parameter of House Price
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#importing data
data = pd.read_csv('Raw_Housing_Prices.csv')
data.head()

data['Sale Price'].describe()

#distribution of target variable
data['Sale Price'].plot.hist()

# checking quantiles
q1 = data['Sale Price'].quantile(0.25)
q3 = data['Sale Price'].quantile(0.75)
q1, q3

#calculating iqr
iqr = q3 - q1
iqr

upper_limit = q3 + 1.5*iqr
lower_limit = q1 - 1.5*iqr
upper_limit, lower_limit

# imputing outliers
def limit_imputer(value):
  if value > upper_limit:
    return upper_limit
  if value < lower_limit:
    return lower_limit
  else:
    return value
   
data['Sale Price'] = data['Sale Price'].apply(limit_imputer)
data['Sale Price'].describe()

data['Sale Price'].plot.hist()

#checking missing values
data.isnull().sum()

data['Sale Price'].dropna(inplace=True)
data["Sale Price"].isnull().sum()

data.info()

#isolating numerical variables
numerical_columns = ['No of Bathrooms', 'Flat Area (in Sqft)','Lot Area (in Sqft)',
                     'Area of the House from Basement (in Sqft)','Latitude',
                     'Longitude','Living Area after Renovation (in Sqft)']
                     
 #imputing missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

data.info()

####zipcode transform

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
data['Zipcode'] = imputer.fit_transform(data['Zipcode'].values.reshape(-1,1))

data['Zipcode'].shape

column = data["Zipcode"].values.reshape(-1,1)
column.shape

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
data['Zipcode'] = imputer.fit_transform(column)

data.info()

###Other transformations

data['No of Times Visited'].unique()


[Uploading Raw_Housing_Prices.csv…]()






    
