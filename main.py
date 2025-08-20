import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#  معرفی کتابخانه های ساخته شده

from extraction import *    

from tranformations import *

from loading import *
#


#  ------------                                              Extarction:

data=extract_from_csv("./data/loans.csv")   # data it is of type data frame in pandas

# NOTIFICATION!!!!!   in .csv  (( NaN ))  is correct  not   nan or NAN !!!
# print(type(data))
print(data)

#-----------------------------------------------------------------------------------------------

#                                                              TRANSFORMATION:

#   پاک کردن رکوردهای تمام صفر
data=drop_records_all_nans(data)
print(data)


# ---------------------------------------------------------------

# پرکردن فیلدهای خالی 
data=fillna(data)
print(data)


#----------------------------------------------------------------------

#  روش دیگر جهت تبدیل فیلدهای غیر عددی چند حالته(حالت های محدود) به کدهای عددی

data=label_encoder(data)
print(data)


# -----------------------------------------------------------------------------------

#    گسسته سازی 

#   می خواهیم ستون های عددی را در رنج ها( بین ها) یی قرار دهیم واطلاعات را بسته بندی وگسسته کنیم

# data=K_Bins_Discretizer(data)
# print(data)


# ----------------------------------------------------------------------------------------------



#                                     حذف داده های پرت از ستون نرخ
# arr1=np.array(data["rate"])
# plt.boxplot(arr1)
# fig=plt.figure()
# plt.show()

data=remove_rate_outliers(data,0.1,10)
print(data)

#-------------------------------------------------------------------
# جذر گیری از ستون نرخ 
data["rate"]=np.sqrt(data["rate"])
print(data)
# ------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

#   حذف ستون های بی فایده ( مثلا در اینجا ستون نام ها)

data=drop_columns(data,["client_id","loan_id","loan_start","loan_end"])
print(data)

# --------------------------------------------------------------------------------------------------------

#                                   نرمال سازی 

# data=min_max_scaler(data,["client_id","loan_type","loan_amount","repaid","loan_id","loan_start","loan_end","rate"])
# print(data)



# ---------------------------------------------------------------------------------------------------
# محاسبه  متغیر جدید 
data["loan_cost"]=calculate_loan_cost(data)
print(data)

#                                           استاندارد سازی

data=standard_scaler(data,["loan_type","loan_amount","repaid","rate", "loan_cost"])
print(data)

#-----------------------------------------------------------------------
#  ساخت تست و آموزش   test  train

df1,df2 = split_dataframe(data , 330)
print("train dataframe:")
print(df1)
print("test dataframe:")
print(df2)


#       پایان  مرحله ترانسفورم
# -----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
#


#--                                               : LOADING مرحله 

load(data,'./data/targetFile.csv')
load(data,'./data/train.csv')
load(data,'./data/test.csv')





#  SUCCESSFULLY!!!!

#          END OF ETL