# week 1, 2
arr[1 : 2]
arr.reshape(3, 3)
np.concatenate(arr1, arr2, axis = 1)
np.stack(arr1, arr2, axis = 1)
np.hsplit(arr, 3)
np.vsplit(arr, 3)

arr.max()
arr.min()
arr.mean()
arr.median()
arr.mode()
np.std(arr)
np.quantile(arr, 0.5)

import cv2
img=cv2.imread("/")
reimg=cv2.resize(img, (500,600))
cropped_img=img[300:900,500:1800]
cv2.imshow("original",img)
cv2.imshow("resized",reimg)
cv2.imshow("cropped image",cropped_img)
cv2.imwrite("cropped cat.jpg",cropped_img)
from PIL import Image
img1 = Image.fromarray(np.flipud(arr))
cv2.imshow("img", np.flipud(img1))

# week 3, 4, 5 & 6
df.drop(index=1)
df.drop('Close', axis=1)
df['Rank'] = df['Profit'].rank()
df[['City'’, 'Profit']].sort_values(by="Profit", ascending=True)
.mean()
.median()
.mode()
.std()
.min()
.max()
.describe()
.info()
.count()
.unique()
df.rename(columns={'Column': 'New Column'}, inplace=True)

# week 7, 8, 9 & 10
df.isnull().sum()
df.dropna()
df.fillna(0)
df.notnull()

df['gender'] = df['name'].map(genders) -> series or dictionary or function
def interview(row):
	return row['age'] < 45 and row['income'] > 75000
df['interview'] = df.apply(interview, axis=1)

sns.boxplot(y=df.AveBedrms)
q1 = np.quantile(df['AveBedrms'],0.25)
q2 = np.quantile(df['AveBedrms'],0.75)
iqr = q2 - q1
range = iqr * 1.5
lower = q1 - range
upper = q2 + range
outliers = ((df['AveBedrms'] < lower) | (df['AveBedrms'] > upper)).sum()
print(outliers)

for feature in housing.feature_names:
    plt.figure(figsize=(10, 6))
    sns.regplot(x=feature, y='Target', data=df[:100], fit_reg=True)
    plt.xlabel(feature)
    plt.ylabel('Target')
    plt.title(f'Regression and Scatter Plot for {feature}')
    plt.show()

from sklearn.datasets import fetch_california_housing
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Target'] = housing.target
model = ols(formula='Target ~ MedInc + HouseAge + AveRooms + AveBedrms + Population + AveOccup + Latitude + Longitude', data=df).fit()
print(model.summary())
print(anova_lm(model))

# week 11, 12, 13 & 14
- line plot -> plt.plot(x, y)
- bar plots -> plt.bar(categories, values)
- histograms -> plt.hist(data, bins=30)
- density plots -> plt.hist(data, bins=30, density=True), sns.kdeplot(data)
- scatter plots -> plt.scatter(x, y, 'g^')

from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_feature_names)
df['target'] = california_housing.target
sns.scatterplot(x='MedInc', y='target', data=df, hue='AveRooms')

np.random.seed(42)
months = [f'Month_{i}' for i in range(1, 11)]
rainfall_data = pd.DataFrame(np.random.rand(10, 10), columns=months)
fig, axes = plt.subplots(3, figsize=(12, 10))
sns.heatmap(rainfall_data, ax=axes[0], cmap='viridis', annot=True, fmt=".2f")
axes[0].set_title('Default Color Scale (Viridis)')
sns.heatmap(rainfall_data, ax=axes[1], cmap='Blues', annot=True, fmt=".2f")
axes[1].set_title('Blues Color Scale')
sns.heatmap(rainfall_data, ax=axes[2], cmap='Reds', annot=True, fmt=".2f")
axes[2].set_title('Reds Color Scale')
plt.show()

for var in california_feature_names:
    plt.figure(figsize=(8, 6))
    plt.bar(df.index[:100], df[var][:100], color='red')
    plt.xlabel('Index')
    plt.ylabel(var)
    plt.title(f'Bar Plot of {var} in California Dataset')
    plt.show()

np.random.seed(42)
skewed_data = np.random.exponential(scale=2, size=1000)
plt.figure(figsize=(8, 5))
plt.hist(skewed_data, bins=30, edgecolor='black')
plt.show()
transformed_data = np.log(skewed_data + 1)
plt.figure(figsize=(8, 5))
plt.hist(transformed_data, bins=30, edgecolor='black', color='green')
plt.show()

np.random.seed(42)
dates = pd.date_range('2023-01-01', '2023-12-31')
sales = np.random.randint(50, 200, size=len(dates))
sales_data = pd.DataFrame({'Date': dates, 'Sales': sales})
sales_data.set_index('Date', inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(sales_data.index, sales_data['Sales'], label='Sales', color='blue')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
np.random.seed(42)
data = np.random.rand(100, 2) * 10
pca = PCA(n_components=1)
reduced_data = pca.fit_transform(data)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1])
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.subplot(1, 2, 2)
plt.scatter(reduced_data, np.zeros_like(reduced_data))
plt.title('Reduced Data using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('')
plt.tight_layout()
plt.show()
