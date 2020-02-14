import pymongo
import json


JSON_PATH  = "train.json"

NOM_DB  = "db_bigData_assurance" 
NOM_COL = "profil_consumer"

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client[NOM_DB]
collection = db[NOM_COL]

with open(JSON_PATH) as f:
    x = collection.insert_many(json.load(f))

for dbi in client.list_databases():
    print(dbi)