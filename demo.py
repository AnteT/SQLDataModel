from sqlmodel import SQLModel

wc_headers = ['team','rank','federation']
wc_data = [
  ['Argentina',3,'CONMEBOL']
 ,['Brazil',1,'CONMEBOL']
 ,['Ecuador',44,'CONMEBOL']
 ,['Uruguay',14,'CONMEBOL']
 ,['Belgium', 2, 'UEFA']
 ,['Croatia', 12, 'UEFA']
 ,['Denmark', 10, 'UEFA']
 ,['England', 5, 'UEFA']
 ,['France', 4, 'UEFA']
 ,['Germany', 11, 'UEFA']
 ,['Netherlands', 8, 'UEFA']
 ,['Poland', 26, 'UEFA']
 ,['Portugal', 9, 'UEFA']
 ,['Serbia', 21, 'UEFA']
 ,['Spain', 7, 'UEFA']
 ,['Switzerland', 15, 'UEFA']
 ,['Wales', 19, 'UEFA']
 ]

sm = SQLModel(wc_data,wc_headers)
print(sm.colorful("#a6d7e8"))
print(sm.colorful("#b39cf1"))
print(sm.group_by('federation').colorful("#dba4a4"))
sm.get_rows_at_index_range(1,4)
for row in sm.iter_rows():
    print(row)
