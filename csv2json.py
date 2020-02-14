import csv, json

csvPath  = "train.csv"
jsonPath = "train.json"

csvfile = open(csvPath, 'r',encoding = "ISO-8859-1")
jsonfile = open(jsonPath, 'w')


fieldnames = ()

readerFieldName = csv.reader( csvfile, fieldnames)

setFileNames = False
for row in readerFieldName:
    if (not setFileNames):
        setFileNames = True
        fieldnames = tuple(row)
        break
    

reader = csv.DictReader(csvfile, fieldnames)
    
out = json.dumps( [ row for row in reader ] )
print(len(out))
jsonfile.write(out)
print("ok")


# la suite py mongo
