# AICTE-INTERNSHIP-REPORT

## NAME: ABHISHEK KUMAR
## COMPANY: EDUNET FOUNDATION
## INTERNSHIP ID: INTERNSHIP_173070615967287aef12823
## DOMAIN: AICTE INTERNSHIP ON AI BY TECHSAKSHAM INITIATIVE OF MICROSOFT AND SAP
## DURATION: 16 DECEMBER 2024 TO 16 JANUARY 2025
## MENTOR: JAY RATHOD

# OVERVIEW OF THE PROJECT
## PROJECT: Identifying Shopping Trends using Data Analysis
### OBJECTIVE:  
The objective is to develop a machine learning model to predict customer purchase behavior based on a retail dataset. The study addresses the problem of identifying patterns and trends to aid business decisions, such as targeted marketing and inventory management.
The methodology includes data preprocessing, exploratory data analysis (EDA), and the implementation of classification models. Key results show that the Random Forest Classifier achieved the highest accuracy of 92%. Future work will involve integrating additional data sources and exploring advanced models for improved predictions.

### Software Requirements:
•	Python 3.x

•	Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

•	Jupyter notebook

•	Power BI

### Python library code:
In [1]:

1 import pandas as pd

2

import numpy as np

3

import matplotlib.pyplot as plt

4 import seaborn as sns

5 from wordcloud import WordCloud

6 import warnings

7

8

warnings.filterwarnings ("ignore")

9

10

colors = ["#89CFF0", "#FF69B4", "#FFD700", "#7B68EE", "#FF4500", "#9370DB", "#32CD32", "#8A2BE2", "#FF6347", "#20B2AA", "#FF69B4" "#00CED1","#FF7F50", "#7FFF00", "#DA70D6"]I
Customer ID - Unique identifier for each customer.

Age Age of the customer.

Gender - Gender of the customer (Male/Female).

Item Purchased - The item purchased by the customer.

Category - Category of the item purchased.

Purchase Amount (USD) - The amount of the purchase in USD.

Location - Location where the purchase was made.

Size Size of the purchased item.

Color - Color of the purchased item.

Season - Season during which the purchase was made.

Review Rating - Rating given by the customer for the purchased item.

Subscription Status - Indicates if the customer has a subscription (Yes/No).

Shipping Type - Type of shipping chosen by the customer.

Discount Applied - Indicates if a discount was applied to the purchase (Yes/No).

Promo Code Used - Indicates if a promo code was used for the purchase (Yes/No).

Previous Purchases - Number of previous purchases made by the customer.

Payment Method - Customer's most preferred payment method.

Frequency of Purchases - Frequency at which the customer makes purchases (e.g., Weekly, Fortnightly, Monthly)
1

2

df = pd.read_csv("./shopping_trends.csv")

3

4

# df.shape, df.columns, df.info(), df.describe()

5

# df.isnull().sum(), df.duplicated().sum()

6

df.sample(5)
In [3]:

1 ax= df['Gender').value_counts().plot(kind='bar', rot=0,color=colors)

2 plt.show()

1 counts = df ["Gender"].value_counts()

2 counts.plot(kind='pie', colors = colors, explode=(0,0.1), autopct='%1.1f%%')

3 plt.xlabel('Gender', weight='bold')

4 plt.legend()

5 plt.show()
In [5]:

1 sns.histplot(data=df ['Age'), color='skyblue')

2

plt.show()
In [6]:

1

df["Category"].value_counts()

Out [6]:

Category

Clothing

1737

Accessories

1240

Footwear

599

Outerwear

324

Name: count, dtype:

int64

In [7]:

1 ax = df["Category").value_counts().plot(kind = 'bar', color = colors, rot = 0)

2

3

4

5

for pin ax.patches:

ax.annotate(int(p.get_height()), (p.get_x()+0.25,p.get_height()+1),ha='center')

6

plt.xlabel('Categories', weight = "bold")

7 plt.ylabel('Number of Occurances', weight='bold')

8

plt.show()
1 plt.figure(figsize = (20, 6))

2 counts = df ["Category"].value_counts()

3 explode=[0.1]*len (counts)

4 counts.plot(kind='pie', colors = colors, explode=explode, autopct='%1.1f%%')

5 plt.xlabel('Category', weight='bold')

6 plt.legend()

7 plt.show()
1 def get_pieChart(column):

2 plt.figure(figsize = (20, 6))

3 counts = df[column].value_counts()

4 explode=[0]*(len(counts)-1) + [0.1]

5

counts.plot(kind='pie', colors = colors, explode=explode, autopct='%1.1f%%')

6 plt.xlabel(column, weight='bold')

7 plt.legend()

8 plt.show()

In [10]:

1 get_pieChart("Subscription Status")
In [11]:

T

1 get_pieChart("Payment Method")
7

8

9

def getBarChart(column):

plt.figure(figsize = (20,6))

ax = df [column].value_counts().plot(kind = 'bar', color = colors, rot = 0)

for p in ax.patches:

ax.annotate(int(p.get_height()), (p.get_x()+0.25,p.get_height()+1),ha='center')

plt.xlabel(column, weight = "bold")

plt.ylabel('Number of Occurances', weight='bold')

10

plt.show()

In [13]:

1 getBarChart('Payment Method')

In [14]:

1

getBarChart ('Shipping Type')
In [15]:

1 getBarChart ('Item Purchased')
1

plt.figure(figsize=(16,8))

2 df['Item Purchased'].value_counts().sort_values().plot(kind='barh',color=sns.color_palette("tab10"),

3

4

edgecolor='black')

I

5 plt.ylabel('Item Purchased', fontsize = 16)

6 plt.xlabel('\nNumber of Occurrences', fontsize = 16)

7 plt.title('Item Purchased\n', fontsize = 16)

8

9 plt.show()
In [17]:

1 df["Location"].value_counts(
In [18]:

1

df["Size"].value_counts(

1755

Out [18]:

Size

M

L

1053

S

XL

663

429

Name: count, dtype: int64
In [19]:

1 df['Category'].value_counts()

2 df["Color"].value_counts()

3 df ["Season"].value_counts()

Out [19]:

Season

Spring 999

Fall

975

Winter

971

Summer 955

Name: count, dtype: int64

In [20]:

1 text = "".join(title for title in df ["Frequency of Purchases"])

2 word_cloud = WordCloud (collocations = False, background_color = 'white').generate (text)

3 plt.axis("off")

I

4 plt.imshow(word_cloud)

5 plt.show()
In [22]:

1

# What is the average age of customers in the dataset ?

2

average_age = df['Age'].mean()

3 print("Average Age:", average_age)

Average Age: 44.06846153846154

In [23]:

1

What is the most common item purchased ?

2

df['Item Purchased'].mode()

3

df['Item Purchased'].mode()[0]

Out [23]:

'Blouse'

In [24]:

1

df[df['Gender']=='Male'] ['Item Purchased'].mode()

Out [24]:

0 Pants

Name: Item Purchased, dtype: object
In [25]:

1

df[df['Gender']=='Female'] ['Item Purchased'].mode()

Out [25]:

0

Blouse

Name:

Item Purchased, dtype: object

In [26]:

1

# What is the most common season for purchases ?

2

most_common_season = df['Season'].mode()[0]

3 print("Most Common Season for Purchases:", most_common_season)

Most Common Season for Purchases:

Spring

In [27]:

1 # What is the maximum and minimum review rating in the dataset ?

2

max_review_rating = df ['Review Rating'].max()

3

min_review_rating = df ['Review Rating'].min()

4

print("Maximum Review Rating:", max_review_rating)

5

print("Minimum Review Rating:", min_review_rating)

Maximum Review Rating: 5.0

Minimum Review Rating: 2.5
In [28]:

1 # What is the average review rating

2 # for male customers and female customers separately?

3 average_rating_male = df [df['Gender'] == 'Male'] ['Review Rating'].mean()

4 average_rating_female = df [df['Gender'] == 'Female'] ['Review Rating'].mean()

5 print("Average Review Rating for Male Customers:", average_rating_male)

6 print("Average Review Rating for Female Customers:", average_rating_female)

Average Review Rating for Male Customers: 3.7539592760180995

Average Review Rating for Female Customers: 3.741426282051282

In [29]:

1 # What is the most common category of items purchased by male customers in the Winter season

2 # with a review rating below 3?

3 common_category_low_rating_male_winter = df[(df['Gender'] == 'Male')

4

5

& (df['Season'] == 'Winter')

& (df['Review Rating'] < 3)] ['Category'].mode()[0]

6 print("Most Common Category for Low-Rating Male Customers in Winter Season:", common_category_low_rating_male_winter)

Most Common Category for Low-Rating Male Customers in Winter Season: Clothing

In [30]:

1 # How many customers have a subscription status of 'Yes'

2 # and used a promo code for their purchase ?

3 subscription_promo_count = df [(df['Subscription Status'] == 'Yes')

4 & (df['Promo Code Used'] == 'Yes')] ['Customer ID'].count()

5 print("Number of Customers with Subscription and Promo Code Used: ", subscription_promo_count) 

### Conclusion:
This project demonstrates the potential of predictive modeling in retail analytics. The Random Forest Classifier outperformed other models with an accuracy of 92%, providing valuable insights for targeted marketing and decision-making.
