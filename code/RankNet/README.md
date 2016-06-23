README
======
This contains our project code for the course Computer Vision.
Code is adapted from https://github.com/shiba24/learning2rank
###Requirements
See above link from
Run main.py to learn the RankNet model. You need features.csv file and corresponding labels.txt file to run the code. Change the path of these files in the code suitably.  
After defining the path suitably, run the command
```
python learning2rank/rank/main.py
```
It will create 10 ranknet models in the directory, one for each attribute.  
To obtain ranking of features using these learnt models, run load.py by making similar changes of file path in it as of main.py
```
python learning2rank/rank/load.py
```
Save generated ranks in rank.txt file.  
To calculate accuracy of the generated rank run acc.py.
```
python acc.py
```
It will use truerank.csv file and nlabel.txt file.  
Use learning2rank/rank/zero.py for zero shot learning.
