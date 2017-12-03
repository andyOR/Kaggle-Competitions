## Kaggle Churn prediction KKBox
import os;
os.chdir('c:\\2017\Work\Python\Kaggle')

## Reading and making data

f = open('train.csv', 'r')
names = f.read()
names_list = names.split('\n')
nested_list = []
for line in names_list:
    comma_list = line.split(',')
    nested_list.append(comma_list)
nested_list = nested_list[1:]

numerical_list = []
for line in nested_list:
    name = line[0]
    count = float(line[1])
    new_list = [name, count]
    numerical_list.append(new_list)
print(numerical_list[0:5])


## getting unique values with dictionaries
def calc_counts(data, column):
    response = {}
    for itr in data:
        ext1 = itr[column]
        if ext1 in response:
            response[ext1] += 1

        else:
            response[ext1] = 1

    return (response)


## Reading transaction data with csv function

import csv
f = open("transactions.csv", 'r')
csv = csv.reader(f)
transact = list(csv)
print(transact[0:5])
transact = transact[1:]

def unique(data, column):
    unique_list = []
    for row in data:
        value = row[column]
        unique_list.append(value)
    p = set(unique_list)
    q = len(p)
    return [p, q]


##Changing the date format of transaction date and membership expiration date

## Unfinished- later        
date_list = []
for row in transact:
    t_date = row[6]
    e_date = row[7]
    parts_1 = .split("-")
    birth_years.append(parts[0])


## Finding the highest frequency and relevent value


def high_value(data, column):
    dict_1 = {}
    for row in data:
        keys = row[column]
        if keys in dict_1:
            dict_1[keys] += 1
        if keys not in dict_1:
            dict_1[keys] = 1
    return dict_1

    max_value = None
    for k,v in dict_1.items():
        if max_value is None or v > max_value:
            max_value = v
            
    max_name = []
    for k,v in dict_1.items():
        if v == max_value:
            max_name.append(k)

    print("Max_repsonse=")
    print(max_name)
    print("\nMax Value=")
    print(max_value)
        

## Finding specific number in dates or any other column

     days = 0
    for row in transact:
        p=row[7]
        if re.search("0416", p)is not None:
            days+=1
    days = 0
    for row in g:
        p = str(row)
        if p[2:4] == 0:
            days+=1


## Dates Operations


import datetime
date_today = datetime.datetime(year=2017, month=10, day=4)
today = date_today.strftime("%x")
print(today)

def count_posts_in_month(month):
    count = 0
    for row in posts:
        if row[2].month == month:
            count += 1
    return count

feb_count = count_posts_in_month(2)
aug_count = count_posts_in_month(8)

import datetime
date_counts = {}
for row in data:
    dates = datetime.datetime(year = int(row[1]), month = int(row[2]), day =1)
    row.append(dates)
    if dates in date_counts:
        date_counts[dates] = date_counts[dates] + 1
    else:
        date_counts[dates] = 1


## PANDAS
## Pandas reading data

import pandas as pd
transact = pd.read_csv("transactions.csv")
col_names = transact.columns.tolist()
print(col_names)
print(transact.head(3))
transact.dtypes# to know data types of columns


## Row operations on columns
# Looking at diiference in paid amount by subscribers

diff  = transact['plan_list_price']- transact['actual_amount_paid']
set(diff)

dict_1 = {}
for keys in diff:
    if keys in dict_1:
        dict_1[keys] += 1
    if keys not in dict_1:
        dict_1[keys] = 1

# Use series.value_counts() for each reponse total
plan_days = transact["payment_plan_days"]
plan_days.value_counts()


## TO find number of rows and columns
transact.shape



##Finding Missing values in columns

columns = transact.columns.tolist()
def missing_row(data, columns):
    for col in columns:
        col_is_null = pd.isnull(data[col])
        col_null_true = data[col_is_null]
        col_null_count = len(col_null_true)
        print(col)
        print("=")
        print(col_null_count)


## Pivot function

churn_diff = transact.pivot_table(index= 'is_cancel', values = 'plan_list_price')
print(churn_diff)

# There is difference in mean of amount paid for ppl with canceled subsription
churn_diff_paid = transact.pivot_table(index= 'is_cancel', values = 'actual_amount_paid')
print(churn_diff_paid)

           actual_amount_paid
is_cancel
0                  143.455783
1                  103.202970

## Advanced Pivot

import numpy as np
port_stats = transact.pivot_table(index="is_cancel", values=["plan_list_price", "actual_amount_paid"], aggfunc=np.mean)
print(port_stats)



## Checking for any reduction in information due to dropping of rows

transact_drop = transact.dropna(axis=0)
len(transact)
len(transact_drop)

# Only dropping rows in specific column

transact_drop = transact.dropna(axis=0, subset= ["plan_list_price", "actual_amount_paid"])


## Sorting on dates and resetting the index

transact_sort = transact.sort_values('transaction_date')
transact_sort.reset_index(drop=False)


## Dataframe.apply

def auto_renew_cancel(row):
    if row["is_auto_renew"]== 1:
        return "yes"
    else:
        return "no"

transact["auto_renew"] = transact.apply(auto_renew_cancel, axis =1)

#data on auto_renew member canceling the subscription

auto_renew_cancel = transact.pivot_table(index='auto_renew', values = 'is_cancel')
plan_list_cancel = transact.pivot_table(index = "plan_list_price", values="is_cancel")
payment_plan_cancel = transact.pivot_table(index = "payment_plan_days", values="is_cancel")


# Unique values in payment column.
print(transact['payment_plan_days'].unique())

col_counts = dict()

def calculate_col_totals(df, col):
    cats = df[col].unique()
    counts_dictionary = dict()

    for c in cats:
        major_df = df[df[col] == c]
        total = major_df[col].sum()
        counts_dictionary[c] = total
    return counts_dictionary

col_counts = calculate_col_totals(transact)

##Using string indexes to find variables
# Import the Series object from pandas
from pandas import Series

series_payment = transact["payment_method_id"]
series_msno = transact["msno"]

msno_names = series_msno.values
plan_scores = series_payment.values
series_custom = Series(plan_scores , index=msno_names)
series_custom[['AZtu6Wl0gPojrEQYB8Q3vBSmE2wnZ3hi1FbK1rQQ0A4=', '8qrtRZQTuCih4YJhjEwvVdi9ojgltQnW5Rmqz3iMRXU=']]


## Sorting index on column
import pandas as pd
transact = pd.read_csv('transactions.csv')
transact_price = transact.set_index('plan_list_price', drop=False)
print(transact_price.index)

plan_price = [129, 149]
List_129 = transact_price[plan_price]


## Using apply method

import numpy as np
types = transact.dtypes
# filter data types to just floats, index attributes returns just column names
float_columns = types[types.values == 'int64'].index
# use bracket notation to filter columns to just float columns
float_df = transact[float_columns]
plan_list_pp = transact.columns
plan_list_list_price = plan_list_pp[["plan_list_price", "actual_amount_paid"]]
plan_paid = plan_list_price_paid.apply(lambda x: np.diff(x), axis = 1)

## Finding row number for error flag

for i, date in enumerate(g):
   try:
      date = str(date)
      p = date[2:4]
      q = date[4:6]
      r = date[6:8]
      dates = datetime.datetime(year = int(p), month = int(q), day =int(r))
      date= dates.strftime("%x") # write your function here
   except ValueError:
      print('ERROR at index {}: {!r}'.format(i, date))

for i, date in enumerate(t):
   try:
     date = int(str(date)
   except ValueError:
      date = "NAN"


# Kaggle Competition
## Reading datasets

import pandas as pd
transact = pd.read_csv("transactions.csv")
members = pd.read_csv("members.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("sample_submission_zero.csv")

import pandas as pd
train_tran_mem = pd.read_csv("train_tran_mem.csv")
test_tran_mem = pd.read_csv("test_tran_mem.csv")


## 
import pandas as pd
train_tran_mem_user = pd.read_csv("train_tran_mem_user.csv")
test_tran_mem_user = pd.read_csv("test_tran_mem_user.csv")


## Making csv
df.pd.to_csv("file.csv")

## Reforming dates column

from pandas import Series
import datetime


def dates_reform(date):
    date = str(date)
    p = date[2:4]
    q = date[4:6]
    r = date[6:8]
    dates = datetime.datetime(year = int(p), month = int(q), day =int(r))
    today = dates.strftime("%x")
    return today
    return row.name

transact["transact_conv_date"] = transact["transaction_date"].apply(dates_reform)
transact["transact_mem_exp_date"] = transact["membership_expire_date"].apply(dates_reform)
members["regis_time"]=members["registration_init_time"].apply(dates_reform)
members["expir_time"]=members["expiration_date"].apply(dates_reform)


## Selecting unique numerical columns
train_tran1 = train_tran1.select_dtypes(include=['int64','float64'])
test_tran1 = test_tran1.select_dtypes(include=['int64','float64'])


## dropping unnecessary columns
train_tran1 = train_tran1.drop(['regis_time','expir_time', 'transact_conv_date','transact_mem_exp_date'], axis=1)
test_tran1 = test_tran1.drop(['regis_time','expir_time', 'transact_conv_date','transact_mem_exp_date'], axis=1)
train_tran1 = train_tran1.drop(['regis_exp_date','expiration_date'],axis=1)
test_tran1 = test_tran1.drop(['regis_exp_date','expiration_date'],axis=1)


#Converting object into date types
transact1 = transact[0:10]
transact["transact_conv_date"] = pd.to_datetime(transact["transact_conv_date"])
transact["transact_mem_exp_date"] = pd.to_datetime(transact["transact_mem_exp_date"])
members["regis_time"] = pd.to_datetime(members["regis_time"])
members["expir_time"] = pd.to_datetime(members["expir_time"])
transact["Membership_se_days"]=[int(i.days) for i in (transact["transact_mem_exp_date"] - transact["transact_conv_date"])]
members["regis_exp_date"]= [int(i.days) for i in (members["expir_time"] - members["regis_time"])]



train_tran = train.merge(transact, on="msno", how = "left")# merged on train msno
test_tran = test.merge(transact, on="msno", how = "left")
train_tran_mem = train_tran.merge(members, on="msno", how = "left")# merged on train msno
test_tran_mem = test_tran.merge(members, on="msno", how = "left")
train_tran_mem_user = train_tran_mem.merge(users1, on="msno", how = "left")
test_tran_mem_user = test_tran_mem.merge(users1, on="msno", how ="left")

train_tran1 = train_tran_mem.drop_duplicates(keep="first", subset = ["msno"])# dropped duplicated column from transaction
test_tran1 = test_tran_mem.drop_duplicates(keep="first", subset = ["msno"])
train_tran1.reset_index(inplace=False)
test_tran1.reset_index(inplace=False)

## duplicates dropping

train_tran1 = train_tran_mem_user.drop_duplicates(keep="first", subset = ["msno"])# dropped duplicated column from transaction
test_tran1 = test_tran_mem_user.drop_duplicates(keep="first", subset = ["msno"])
train_tran1.reset_index(inplace=False)
test_tran1.reset_index(inplace=False)
#train_t = train_tran1.fillna(0)
#test_t = test_tran1.fillna(0)

## Relations between columns with pivot functions
pay = train_tran1.pivot_table(index = "membership_se_days", values="is_churn")

train_tran1["payment_method_id"] = train_tran1["payment_method_id"].fillna(method='pad')
train_tran1["payment_method_id"] = train_tran1["payment_method_id"].fillna(41)
train_tran1["payment_plan_days"] = train_tran1["payment_plan_days"].fillna(method='pad')
train_tran1["payment_plan_days"] = train_tran1["payment_plan_days"].fillna(30)
train_tran1["plan_list_price"] = train_tran1["plan_list_price"].fillna(method='pad')
train_tran1["plan_list_price"] = train_tran1["plan_list_price"].fillna(149)
train_tran1["is_auto_renew"] = train_tran1["is_auto_renew"].fillna(1)
train_tran1["Membership_se_days"] = train_tran1["Membership_se_days"].fillna(method='pad')
train_tran1["Membership_se_days"] = train_tran1["Membership_se_days"].fillna(31)
train_tran1["city"] = train_tran1["city"].fillna(method='pad')
train_tran1["city"] = train_tran1["city"].fillna(1)
train_tran1["is_cancel"] = train_tran1["is_cancel"].fillna(method='pad')
train_tran1["is_cancel"] = train_tran1["is_cancel"].fillna(0)
train_tran1["actual_amount_paid"] = train_tran1["actual_amount_paid"].fillna(method='pad')
train_tran1["actual_amount_paid"] = train_tran1["actual_amount_paid"].fillna(136)

#column bd
train_tran1.loc[train_tran1['bd'] < 0, 'bd'] = 23
train_tran1.loc[train_tran1['bd'] > 80, 'bd'] = 23
train_tran1.loc[train_tran1['bd'] == 0, 'bd'] = 23
train_tran1["bd"]=train_tran1["bd"].fillna(method='pad')
train_tran1["bd"]=train_tran1["bd"].fillna(23)

## column gender
train_tran1['gender'] = train_tran1['gender'].map({'female': 2, 'male': 1})
train_tran1["gender"] = train_tran1["gender"].fillna(2)
train_tran1["regis_exp_date"] = train_tran1["regis_exp_date"].fillna(method='pad')
train_tran1["regis_exp_date"] = train_tran1["regis_exp_date"].fillna(1471)
train_tran1["date_count"] = train_tran1["date_count"].fillna(method="pad")
train_tran1["num_25"] = train_tran1["num_25"].fillna(method='pad')
train_tran1["num_50"] = train_tran1["num_50"].fillna(method='pad')
train_tran1["num_75"] = train_tran1["num_75"].fillna(method='pad')
train_tran1["num_985"] = train_tran1["num_985"].fillna(method='pad')
train_tran1["num_100"] = train_tran1["num_100"].fillna(method='pad')
train_tran1["num_unq"] = train_tran1["num_unq"].fillna(method='pad')
train_tran1["total_secs"] = train_tran1["total_secs"].fillna(method='pad')




test_tran1["payment_method_id"] = test_tran1["payment_method_id"].fillna(method='pad')
test_tran1["payment_method_id"] = test_tran1["payment_method_id"].fillna(41)
test_tran1["payment_plan_days"] = test_tran1["payment_plan_days"].fillna(method='pad')
test_tran1["payment_plan_days"] = test_tran1["payment_plan_days"].fillna(30)
test_tran1["plan_list_price"] = test_tran1["plan_list_price"].fillna(method='pad')
test_tran1["plan_list_price"] = test_tran1["plan_list_price"].fillna(149)
test_tran1["is_auto_renew"] = test_tran1["is_auto_renew"].fillna(1)
test_tran1["Membership_se_days"] = test_tran1["Membership_se_days"].fillna(method='pad')
test_tran1["Membership_se_days"] = test_tran1["Membership_se_days"].fillna(31)
test_tran1["city"] = test_tran1["city"].fillna(method='pad')
test_tran1["city"] = test_tran1["city"].fillna(1)
test_tran1["is_cancel"] = test_tran1["is_cancel"].fillna(method='pad')
test_tran1["is_cancel"] = test_tran1["is_cancel"].fillna(0)
test_tran1["actual_amount_paid"] = test_tran1["actual_amount_paid"].fillna(method='pad')
test_tran1["actual_amount_paid"] = test_tran1["actual_amount_paid"].fillna(136)


#column bd
test_tran1.loc[test_tran1['bd'] < 0, 'bd'] = 23
test_tran1.loc[test_tran1['bd'] > 80, 'bd'] = 23
test_tran1.loc[test_tran1['bd'] == 0, 'bd'] = 23
test_tran1["bd"]=test_tran1["bd"].fillna(method='pad')
test_tran1["bd"]=test_tran1["bd"].fillna(23)

## column gender
test_tran1['gender'] = test_tran1['gender'].map({'female': 2, 'male': 1})
test_tran1["gender"] = test_tran1["gender"].fillna(2)
test_tran1["regis_exp_date"] = test_tran1["regis_exp_date"].fillna(method='pad')
test_tran1["regis_exp_date"] = test_tran1["regis_exp_date"].fillna(1464)
test_tran1["date_count"] = test_tran1["date_count"].fillna(method="pad")
test_tran1["num_25"] = test_tran1["num_25"].fillna(method='pad')
test_tran1["num_50"] = test_tran1["num_50"].fillna(method='pad')
test_tran1["num_75"] = test_tran1["num_75"].fillna(method='pad')
test_tran1["num_985"] = test_tran1["num_985"].fillna(method='pad')
test_tran1["num_100"] = test_tran1["num_100"].fillna(method='pad')
test_tran1["num_unq"] = test_tran1["num_unq"].fillna(method='pad')
test_tran1["total_secs"] = test_tran1["total_secs"].fillna(method='pad')


## Discount column
train_tran1["discount"]= [int(i) for i in (train_tran1["plan_list_price"] - train_tran1["actual_amount_paid"])]
train_tran1.loc[train_tran1['discount'] < 0, 'discount'] = 0
test_tran1["discount"]= [int(i) for i in (test_tran1["plan_list_price"] - test_tran1["actual_amount_paid"])]
test_tran1.loc[test_tran1['discount'] < 0, 'discount'] = 0
discount_churn = train_tran1.pivot_table(index = "is_churn", values="discount")

d_churn = train_tran1.pivot_table(index = "is_churn", values="num_25")


## Dummy variables for categorical variables
dummy_payment = pd.get_dummies(train_tran1["payment_method_id"], prefix="paymentid")
train_tran1 = pd.concat([train_tran1, dummy_payment], axis=1)

dummy_payment = pd.get_dummies(test_tran1["payment_method_id"], prefix="paymentid")
test_tran1 = pd.concat([test_tran1, dummy_payment], axis=1)


## using non-missing columns
null_series = train_tran1.isnull().sum()
series = null_series[null_series == 0]
print(series)

predictors = series.index
predictors = predictors.tolist()
#predictors.remove("is_churn")
predictors.remove("msno")

df = train_tran1[predictors]


## normalizing dataframe
df_norm = (df - df.mean()) / (df.max() - df.min())
train_tran = df_norm
train_tran["is_churn"] = train_tran1["is_churn"]
train_tran["msno"] = train_tran1["msno"]



df = test_tran1[predictors]
df_norm = (df - df.mean()) / (df.max() - df.min())
test_tran = df_norm
test_tran["msno"] = test_tran1["msno"]







train_subset = train_tran1[series.index]
corrmat = train_subset.corr()
sorted_corrs = corrmat['is_churn'].abs().sort_values()
strong_corrs = sorted_corrs[sorted_corrs > 0.04]
print(strong_corrs)

## Finding correlation
import seaborn as sns
import matplotlib.pyplot as plt 
plt.figure(figsize=(10,6))
strong_corrs = sorted_corrs[sorted_corrs > 0.04]
corrmat = train_tran1[strong_corrs.index]
axh = sns.heatmap(corrmat)

## Dropping correlated columns

final_corr_cols = strong_corrs.drop(['num_25', 'actual_amount_paid'])
predictors = final_corr_cols.drop(['is_churn']).index

## High variance features
unit_train = train_tran1[features]/(train_tran1[features].max())
sorted_vars = unit_train.var().sort_values()
print(sorted_vars)


## Keeping columns with high variance and dropping rest
predictors = features.drop(['payment_plan_days','plan_list_price','payment_method_id'])



# Randomizing the traain data

import sklearn.utils
train_random = sklearn.utils.shuffle(train_tran1)
#print('\n\ntrain_random: {0}'.format(train_random))
train_data = train_random.reset_index(drop=True)


## MACHINE LEARNING SUBMISSION

train = train_data[1:600000]
test = train_data[600001:]


train = X1[1:600001]
test = X1[600001:]
# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross-validation
from sklearn.model_selection import KFold

# The columns we'll use to predict the target
predictors = ['payment_plan_days','plan_list_price', "is_cancel", "is_auto_renew", 'city', ]
predictors = ["Membership_se_days"]

# Initialize our algorithm class
alg = LinearRegression()

# It returns the row indices corresponding to train and test
# We set random_state to ensure we get the same splits every time we run this
kf = KFold(3, random_state=1)

predictions = []
for train, test in kf.split(titanic):
        # The predictors we're using to train the algorithm  
    # Note how we only take the rows in the train folds
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# Initialize our algorithm
lr = LogisticRegression()
# Compute the accuracy score 
scores = cross_val_score(alg, train_data[predictors], train_data["is_churn"], cv=3)
# Take the mean of the scores
print(scores.mean())

##
alg.fit( train_1[predictors], train_1["is_churn"])
train_prediction = alg.predict(train_2[predictors])
accuracy = len(train_prediction[train_prediction == train_2["is_churn"]]) / len(train_prediction)
print(accuracy)

alg.fit( train_data[predictors], train_data["is_churn"])
train_prediction = alg.predict(test_tran1[predictors])

submission = pd.DataFrame({
    "is_churn": test_predictions,
        "msno": test_tran1["msno"]        
    })

submission.to_csv("sample_submission13.csv", index=False)



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
alg = LinearRegression()
kf = KFold(5, random_state=1)

predictions = []
for train, test in kf.split(train_data):
        # The predictors we're using to train the algorithm  
    # Note how we only take the rows in the train folds
    train_predictors = (train_data[predictors].iloc[train,:])
    # The target we're using to train the algorithm
    train_target = train_data["is_churn"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(train_data[predictors].iloc[test,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

### Map predictions to outcomes (the only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

accuracy = len(predictions[predictions == train_data["is_churn"]]) / len(predictions)

## 

lr.fit(train_data[predictors], train_data['is_churn'])
test_predictions = lr.predict(test_tran1[predictors])


##

lr.fit(X1[predictors], X1['is_churn'])
test_predictions = lr.predict(Y1[predictors])

##train_predictions[train_predictions > .10] = 1
##train_predictions[train_predictions <=.10] = 0
##test_predictions[test_predictions > 0.10] = 1
##test_predictions[test_predictions <=0.10] = 0

## Choosing only float columns
df.select_dtypes(include=['float64'])


predictors = ['payment_plan_days','plan_list_price', "is_cancel", "is_auto_renew", 'city',"payment_method_id", "Membership_se_days","actual_amount_paid", "bd", "gender","regis_exp_date"]
predictors = ["is_cancel","is_auto_renew", "city", "payment_plan_days", "plan_list_price", "payment_method_id"]
predictors = ["Membership_se_days","regis_exp_date","plan_list_price", "actual_amount_paid", "city", "gender", "is_cancel","is_auto_renew","plan_list_price",'payment_plan_days','num_25', 'num_50', 'num_75','num_985','num_100','total_secs']

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[predictors], train['is_churn'])
from sklearn.metrics import mean_squared_error
train_predictions = lr.predict(train[predictors])
test_predictions = lr.predict(test[predictors])



train_mse = mean_squared_error(train_predictions, train['is_churn'])
test_mse = mean_squared_error(test_predictions, test['is_churn'])
import numpy as np
#train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
#print(train_rmse)
print(test_rmse)


## Decision trees from dataquest

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

columns = ["Membership_se_days","regis_exp_date","plan_list_price", "actual_amount_paid", "city", "gender", "is_cancel","is_auto_renew","plan_list_price",'payment_plan_days','num_25', 'num_50', 'num_75','num_985','num_100','total_secs']

clf = DecisionTreeClassifier(random_state=1, max_depth = 7, min_samples_split = 13)
clf.fit(train[columns], train["is_churn"])
predictions = clf.predict(test[columns])
test_auc = roc_auc_score(test["is_churn"], predictions)

train_predictions = clf.predict(train[columns])
train_auc = roc_auc_score(train["is_churn"], train_predictions)

print(test_auc)
print(train_auc)
