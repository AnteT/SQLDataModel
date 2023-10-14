from SQLDataModel_future import SQLDataModelFuture
import pandas

headers = ['country','region','check','total','report date']
data = (
     ('US','West','Yes',2016,'2023-08-23 13:11:43')
    ,('US','West','No',1996,'2023-08-23 13:11:43')
    ,('US','West','Yes',1296,'2023-08-23 13:11:43')
    ,('US','West','No',2392,'2023-08-23 13:11:43')
    ,('US','Northeast','Yes',1233,'2023-08-23 13:11:43')
    ,('US','Northeast','No',3177,'2023-08-23 13:11:43')
    ,('US','Midwest','Yes',1200,'2023-08-23 13:11:43')
    ,('US','Midwest','No',2749,'2023-08-23 13:11:43')
    ,('US','Midwest','Yes',1551,'2023-08-23 13:11:43')
)

sdm_future = SQLDataModelFuture(data,headers)
print(sdm_future)
sdm_future = sdm_future[2:5]
result_df = sdm_future.to_pandas()
print(result_df)
from_pandas_sdm = SQLDataModelFuture.from_pandas(result_df)
print(from_pandas_sdm)