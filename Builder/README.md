Tutoriel pour executer le fichier builder.py

Préréquis:=========================================================================

-Installer Python (version>=3.7 pour PyC3Dserver)

-Installer PyC3Dserver <pip install pyc3dserver> (plus d'infos: https://github.com/mkjung99/pyc3dserver)

-Installer les bibliothèques sys, pandas, math et numpy

-Avoir un fichier qtm avec Qualisys


Exporter qtm vers c3d dans Qualisys===============================================

-Depuis "Unidentified trajectories", mettre les trajectoires avec un fill level supérieur à 1% dans "Labeled trajectories". Il ne fait pas avoir plus de 255 trajectoires dans "Labeled trajectories", il faut augmenter le seuil des 1% pour avoir moins de trajectoires.

-Cliquer sur File->Export->To C3D... Les cases "exclude unidentified trajectories" et "exclude empty trajectories" doivent êtres cochées, Label Format: De facto standard, Event Outout Format: Following the c3d specification, Units: Millimeters. Cliquer sur "OK"

Executer le fichier python:============================================================

-Mettre le ou les fichiers c3d en paramètre: exemple: python builder.py Measurement05.c3d Measurement10.c3d

Pour chaque fichier "file.c3d" en paramètre va etre créé un fichier "file_build.c3d"
Un fichier "Abstract.txt" est créé avec les informations sur les nouveaux fichiers

Mettre un fichier créé dans qualisys:
	Supprimer le marqueur "a_supprimer" de "Labeled trajectories"
