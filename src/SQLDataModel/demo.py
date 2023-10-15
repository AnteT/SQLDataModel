from SQLDataModel import SQLDataModel

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

sdm = SQLDataModel(data,headers)


print(sdm) # prints standard color
print(sdm.colorful("#a6d7e8")) # outputs as teal
print(sdm.colorful("#b39cf1")) # outputs as purple

sdm_grouped = sdm.group_by('region','check') # group by one or more columns
print(sdm_grouped.colorful("#dba4a4"))

sdm_grouped.to_csv('model-output.csv',include_index=True) # save as csv, sqlite, pickle, text
sdm = sdm.get_rows_at_index_range(1,4) # slice by method or using python slice indexing [1:4]

for row in sdm.iter_rows(): # iterate over data
    print(row)
