# BigDataProject

## Listes des fichiers

- *encriptionFile* : **Fonctions de chiffrement et déchiffrement via KMS d'AWS**
- *predict.py* : **Fonctions permettant d'appliquer le modèlegal**



- *send_file.py* : **Partie permettant d'envoyer les fichiers sur le bucket et de d'envoyer un message à la SQS**
- *server.py* : **Partie s'executant sur le server EC2, utilise encryption et predict pour réaliser la prédiction de manière automatique**
- *receive_file.py* : **Partie permettant de recevoir les fichiers du bucket**
- *csv2json.py* : **Partie parmettant de convertir le fichier csv en json**
- *bigData_csv2mongoDB.py* : **Partie s'executant sur la vm permettant de stoquer les resultats obtenus sur la DB mongo, utilise encryption et predict pour réaliser la prédiction de manière automatique**



- *DataAnalysis.ipynb* : **Partie permettant tester les modèles, puis d'exporter le meilleur (best_model.h5)**
- *DataAnalysis.md* : **Partie permettant de visualiser le fichier précedent**
