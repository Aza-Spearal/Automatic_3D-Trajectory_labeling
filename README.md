Le dossier test contient 2 fichiers permettant de tester la détection d'artefacts et les connexions.


Si vous voulez seulement générer un fichier c3d avec les trajectoires assemblées, téléchargez builder.py et template.py.
Si vous voulez inférer des étiquettes avec le réseau de neurones, téléchargez tous le dossier.

Le fichier lstm.py permet d'inférer les étiquettes

Le fichier reader.py permet de lire les données csv et c3d

Le fichier builder.py permet de construire les trajectoires c3d

Le fichier template.c3d permet de générer un fichier c3d avec les trajectoires reconstituées


Le fichier lstm va d'abord lancer le fichier reader.py. Celui ci va lancer le fichier builder.py pour avoir les 15 trajectoires, si les noms des marqueurs ne sont pas donnés par builder.py le programme s'arretera.

Une fois les données c3d obtenues et converties pour servir d'input, reader.py va récupérer les données d'entrainement dans le dossier csv_train. Si les colonnes ne sont pas dans le bon ordre, les données ne seront pas récupérées.


Pour lancer le réseau de neurones, exécuter la commande en mettant le fichier c3d en paramètre: exemple: python lstm.py Measurement12.c3d. Il faut aussi installer les bibliothèques python: torch, random, csv, h5py. Enfin, il faut créer un dossier "csv_train" rempli avec les fichiers csv d'entrainement.

Un fichier Results.txt est généré avec les étiquettes attribuées en fonction des frames.


Que vous utilisiez le fichier builder.py directement ou par l'intermédire du réseau de neurones, voici des consignes à respecter:

Tutoriel pour exécuter le fichier builder.py:

Préréquis====================================================================

-Installer Python (version>=3.7 pour PyC3Dserver)

-Installer PyC3Dserver <pip install pyc3dserver> (plus d'infos: https://github.com/mkjung99/pyc3dserver)

-Installer les bibliothèques python: sys, pandas, math et numpy

-Avoir un fichier qtm avec Qualisys


Exporter qtm vers c3d dans Qualisys==========================================

-Depuis "Unidentified trajectories", mettre les trajectoires avec un fill level supérieur à 1% dans "Labeled trajectories". Il ne fait pas avoir plus de 255 trajectoires dans "Labeled trajectories", il faut augmenter le seuil des 1% pour avoir moins de trajectoires.

-Cliquer sur File->Export->To C3D... Les cases "exclude unidentified trajectories" et "exclude empty trajectories" doivent êtres cochées, Label Format: De facto standard, Event Outout Format: Following the c3d specification, Units: Millimeters. Cliquer sur "OK"

-Si le fichier c3d contient plus de 62576 frames (en 100 fps): créer une copie de ce fichier, supprimer toutes les trajectoires à part une et renommez la pour la retrouver et la supprimer après le traitement. Renommer ce fichier "template.c3d" et mettez le à la place du fichier template.c3d fourni.


Avant l'execution=======================================================

Si les noms des trajectoires commencent toujours avec les mêmes caractères, ceux ci sont supprimés par mon programme pour éviter une redondance inutile. Pour désactiver ça, commentez la section appropriée dans la fonction getData()


Executer le fichier python=======================================================

-Les fichiers builder.py, template.c3d et les fichiers c3d à traiter doivent être dans le même répertoire

-Depuis ce répertoire, exécuter le fichier en mettant le fichier c3d en paramètre: exemple: python builder.py Measurement10.c3d

Pour chaque fichier "file.c3d" en paramètre va etre créé un fichier "file_build.c3d".
Un fichier "Abstract.txt" est créé avec les informations sur les nouveaux fichiers

-Depuis qualisys: Supprimer le marqueur "a_supprimer" de "Labeled trajectories"


Vous pouvez aussi mettre une limite au nombre de connexions si vous voulez garder les connexions les plus probables et éviter trop d'erreur, il faut modifier la variable "limit" dans builder.py
