import pandas as pd 
import random



#  پاک کردن رکوردهای تمام صفر
def drop_records_all_nans(d):
    return d.dropna(how="all",axis=0)   # drops all nan in row axis

    # ----------------------------------------------------------------------

# پرکردن فیلدهای خالی 
def fillna(df):
    max_loan_id=df.loan_id.max()
    df.fillna(value={
        'client_id':df.client_id.mode()[0],
        'loan_type':df.loan_type.mode()[0],
        'loan_amount':df.loan_amount.mean(),
        'repaid' :df.repaid.mode()[0],
        'loan_id':random.choice([i for i in range(max_loan_id) if i not in df["loan_id"].values])  ,
        'loan_start' :df.loan_start.mode()[0],
        'loan_end' :df.loan_end.mode()[0],
        'rate' :df.rate.mean()


    }, inplace=True)
    return df


    # -----------------------------------------------------


#---------------------------------------------

# One_Hot Encoding :


# def One_Hot_Encoder(df):
#     return pd.get_dummies(df,columns=['loan_type'])  # pd.get_dummies gets a data frame and converts the field(column)(which has limited manners: male  and female ) to some new columns 


#----------------------------------------------------------------------

#  روش دیگر جهت تبدیل فیلدهای غیر عددی چند حالته(حالت های محدود) به کدهای عددی


from sklearn.preprocessing import LabelEncoder     #کتابخانه ای برای برخی عملیات مرحله پیش پردازش داده ها
#    python -m pip install scikit-learn

def label_encoder(df):
    le=LabelEncoder()    #  یک ابجکت از کلاس 
    df['loan_type']=le.fit_transform(df['loan_type'])  #   که آبجکت را روی ستون مورد نظر اعمال می کند وخروجی آن یک (( ستون)) است fit_transform  تابعی است 
    #   به هرکدام از حالت های این ستون(نام هر رنگ) یک عدد اختصاص می دهد مثلا آبی صفر قهوه ای 2
    return df


# ----------------------------------------------------------------------------------------------------
#    گسسته سازی 

#   می خواهیم ستون های عددی را در رنج ها( بین ها) یی قرار دهیم واطلاعات را بسته بندی وگسسته کنیم

from sklearn.preprocessing  import KBinsDiscretizer
def K_Bins_Discretizer(df):
    dis=KBinsDiscretizer(n_bins=3 ,encode='ordinal'  , strategy='uniform')
    df['age']=dis.fit_transform(df[['age']])   #  double [[]]  is necessary!!
    return df

# --------------------------------------------------------------------------------------

#  حذف داده های پرت از فیلد وزن

def remove_rate_outliers(df,r_min, r_max):
    df=df[(df['rate']>=r_min) & (df['rate']<=r_max) ]    # in []  is the index of favourate records!!
    return df

#

def split_dataframe(df: pd.DataFrame, row_index):
    """
    df: دیتافریم ورودی
    row_index: شماره سطر مرزی برای تقسیم (از صفر شروع میشه)

    خروجی:
      df1: شامل از ابتدای دیتافریم تا سطر row_index
      df2: شامل از سطر بعدی به بعد
    """
    df1 = df.iloc[:row_index+1]   # تا خود سطر row_index
    df2 = df.iloc[row_index+1:]   # از سطر بعدی به بعد
    
    return df1, df2

#---------------------------------------------------------------------------------------
def drop_columns(data,column_list):
    return data.drop(column_list,axis=1)

# --------------------------------------------------------------------------------------------------------
def split_dataframe(df: pd.DataFrame, row_index: int):
    """
    df: دیتافریم ورودی
    row_index: شماره سطر مرزی برای تقسیم (از صفر شروع میشه)

    خروجی:
      df1: شامل از ابتدای دیتافریم تا سطر row_index
      df2: شامل از سطر بعدی به بعد
    """
    df1 = df.iloc[:row_index+1]   # تا خود سطر row_index
    df2 = df.iloc[row_index+1:]   # از سطر بعدی به بعد
    
    return df1, df2
#  
def calculate_loan_cost(df):
    return df["loan_amount"] * df["rate"]

#                                   نرمال سازی 

from sklearn.preprocessing import MinMaxScaler
def min_max_scaler(df,columns):
    scaler=MinMaxScaler()
    #  بین 0 و1 نرمال می کند
    df=pd.DataFrame(scaler.fit_transform(df))
    # because MinMaxScaler() returns a list of saerated lists not a dataframe, we should convert it to a dataframe by function pd.DataFrame !!!
    # now the titles are deleted so we should recover the titles:
    df.columns=columns   # df.columns  points to titles  
    return df


    # ---------------------------------------------------------------------------------------------------

#                                           استاندارد سازی


from sklearn.preprocessing import StandardScaler
def standard_scaler(df,columns):
    scaler=StandardScaler()
    #  بین 0 و1 نرمال می کند
    df=pd.DataFrame(scaler.fit_transform(df))
    # because MinMaxScaler() returns a list of saerated lists not a dataframe, we should convert it to a dataframe by function pd.DataFrame !!!
    # now the titles are deleted so we should recover the titles:
    df.columns=columns   # df.columns  points to titles  
    return df



