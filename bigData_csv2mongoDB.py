import pymongo
import json
import pandas


JSON_PATH  = "train.json"

NOM_DB  = "db_bigData_assurance"

NOM_COL_INDEX = "profil_consumer_with_index"
NOM_COL = "profil_consumer"

initialised = False
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client[NOM_DB]
collection = db[NOM_COL]
collectionInd = db[NOM_COL_INDEX]
    
if (not initialised):
    print("Init : ")

    #clean old
    collection.drop()
    collectionInd.drop()
    print('Collection clean')

    with open(JSON_PATH) as f:
        all_data = json.load(f)
        x = collection.insert_many(all_data)
        x = collectionInd.insert_many(all_data)

    print('collection created')
    for dbi in client.list_databases():
        print(dbi)

    collectionInd.create_index([ ("Response", 1) ])
    collectionInd.create_index([ ("InsuredInfo_7", 1) ])
    collectionInd.create_index([ ("InsuredInfo_8", 1) ])
    collectionInd.create_index([ ("InsuredInfo_9", 1) ])
    
    print(" - - - - - - - - - - - - - - - - - - - - - - ")

print("Results : ")

print("  -> without Index : ")
print("- Categorie ", collection.find({"Response": 1}).explain()['executionStats']['executionTimeMillis'])
print("- Sexe      ", collection.find({"InsuredInfo_7": 1}).explain()['executionStats']['executionTimeMillis'])
print("- Race      ", collection.find({"InsuredInfo_8": 1}).explain()['executionStats']['executionTimeMillis'])
print("- Religion  ", collection.find({"InsuredInfo_9": 1}).explain()['executionStats']['executionTimeMillis'])

print("")
print("  -> with    Index : ")
print("- Categorie ", collectionInd.find({"Response": 1}).explain()['executionStats']['executionTimeMillis'])
print("- Sexe      ", collectionInd.find({"InsuredInfo_7": 1}).explain()['executionStats']['executionTimeMillis'])
print("- Race      ", collectionInd.find({"InsuredInfo_8": 1}).explain()['executionStats']['executionTimeMillis'])
print("- Religion  ", collectionInd.find({"InsuredInfo_9": 1}).explain()['executionStats']['executionTimeMillis'])


#db.profil_consumer.find({"Response": 1}).explain("executionStats")

#db.profil_consumer.aggregate([{ $group : { _id:"$Response" , num: {$sum : 1}} }])

def count_by_response():
    return db.profil_consumer.aggregate([{ $group : { _id:"$Response" , num: {$sum : 1}} }])

def count_by_sexe():
    return db.profil_consumer.aggregate([{ $group : { _id:"$InsuredInfo_7" , num: {$sum : 1}} }])

def count_by_race():
    return db.profil_consumer.aggregate([{ $group : { _id:"$InsuredInfo_8" , num: {$sum : 1}} }])

def count_by_religion():
    return db.profil_consumer.aggregate([{ $group : { _id:"$InsuredInfo_9" , num: {$sum : 1}} }])

def find_by_response():
    return collection.find({"Response": 1})

def find_by_sexe():
    return collection.find({"InsuredInfo_7": 1})

def find_by_race():
    return collection.find({"InsuredInfo_8": 1})

def find_by_religion():
    return collection.find({"InsuredInfo_9": 1})
