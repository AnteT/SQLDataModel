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
sdm.set_display_color("#b39cf1") # outputs as purple
print(sdm) # prints as purple

sdm_grouped = sdm.group_by(['region','check']) # group by one or more columns
sdm_grouped.set_display_color("#dba4a4")

sdm_grouped.to_csv('model-output.csv',include_index=True) # save as csv, sqlite, pickle, text
sdm = sdm[1:4] # slice by rows and columns [row:rows, col:cols]

for row in sdm.iter_rows(): # iterate over data
    print(row)
