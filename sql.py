import psycopg2
import pandas as pd

conect = psycopg2.connect("host=localhost port=5433 dbname=postgres user=postgres password=###")
cur = conect.cursor()
# cur.execute("INSERT INTO 家計簿 VALUES('2024-04-12','詳細は後で','まだ',0,3000);")
# cur.execute("DELETE FROM 家計簿 WHERE 日付 = '2024-04-12';")
cur.execute(
    "SELECT column_name FROM information_schema.columns WHERE table_name = '家計簿' ORDER BY ordinal_position"
)  # ORDER BY ordinal_positionで順番通りにできる
a = cur.fetchall()
l = []
for oio in a:
    print(oio)
    l.append(oio[0])
print(l)
# conect.commit()  #ここでSQL自体に追加これがないと反映されない
cur.execute("SELECT * FROM 家計簿;")
row = cur.fetchall()
rr = []
for r in row:
    print(r)
    rr.append(r)
df = pd.DataFrame(data=rr, columns=[l])
print(df)
df.to_csv("testpy\\csv_aa.csv")
cur.close()
conect.close()
