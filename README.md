# mass_incomplete

1. Load Data



Baseline： 

mean，mice，HIVAE，GenRBF，RBFnn

Missing rate：5%，10%，20%，30%，40%，50%，60%，70%，80%

Data

| Type | Name | Size | Ord | Nom | Num | link |
| --- | --- | --- | --- | --- | --- | --- |
| cat | CAR| 1728 * 6 | 6 | 0 | 0 | https://archive.ics.uci.edu/dataset/19/car+evaluation |
| cat | BREAST| 286 * 9  | 4 | 5 | 0 | https://archive.ics.uci.edu/dataset/14/breast+cancer |
| mix | AUSTRALIAN | 690 * 14 | 0 | 8 | 6 | https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval |
| mix | HEART | 303 * 13 | 0 | 7 | 6 | https://archive.ics.uci.edu/dataset/45/heart+disease |
| mix  | ADULT | 48842 * 14 | 1 | 7 | 6 | https://archive.ics.uci.edu/dataset/2/adult |
| mix | STUDENT| 649 * 30 | 11 | 16 | 2 | https://archive.ics.uci.edu/dataset/320/student+performance |
| num | BANKNOTE | 1372 * 5 | 0 | 0 | 5 | https://archive.ics.uci.edu/dataset/267/banknote+authentication |
| num | SONAR | 208 * 60  | 0 | 0 | 60 | https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks |
| num | SPAM | 4601 * 57 | 0 | 0 | 57 | https://archive.ics.uci.edu/dataset/94/spambase |
| num | WINE | 4898 * 12 | 0 | 0 | 12 | https://archive.ics.uci.edu/dataset/186/wine+quality |

2. Create Missing Data

- MCAR
- MAR
- MNAR 