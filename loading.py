import pandas as pd 

def load(df,filePath):
    df.to_csv(filePath, index=False)
    #  در انتهای فایل ساخته شده ایندکس نگذار
