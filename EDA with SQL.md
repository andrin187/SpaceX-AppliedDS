# 🚀 SpaceX-AppliedDS
## Explanatory Data Analysis with SQL
SQL queries written to derive questions on variables such as launch sites, payload masses, and dates.

This assignment requires the following SpaceX CSV dataset: [SpaceX Dataset](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv)

Connect to the database...
Load the SQL extension and establish a connection with the database.
```python
!pip install ipython-sql
!pip install ipython-sql prettytable

%load_ext sql

import csv, sqlite3
import prettytable
prettytable.DEFAULT = 'DEFAULT'

con = sqlite3.connect("my_data1.db")
cur = con.cursor()

!pip install -q pandas

%sql sqlite:///my_data1.db

import pandas as pd
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")
```

Remove blank rows from table...
```python
%sql DROP TABLE IF EXISTS SPACEXTABLE;
%sql create table SPACEXTABLE as select * from SPACEXTBL where Date is not null
```

## Analysis
### Task 1
Display the names of the unique launch sites in the space mission:
```python
%sql SELECT DISTINCT launch_site FROM SPACEXTBL;
```

<img width="126" alt="Screenshot 2025-01-31 at 7 37 18 PM" src="https://github.com/user-attachments/assets/c32bc901-16f1-4156-a3ff-6238e054bf8c" />

💡 There are 4 unique launch sites.


### Task 2
Display 5 records where launch sites begin with the string 'CCA':
```python
%sql SELECT * FROM SPACEXTBL WHERE launch_site like 'CCA%' LIMIT 5;
```

<img width="913" alt="Screenshot 2025-01-31 at 7 40 04 PM" src="https://github.com/user-attachments/assets/4d995306-cf54-4b92-8462-9f0156ba2c4d" />

💡 We gain information on specific launch sites starting with CCA.

### Task 3
Display the total payload mass carried by boosters launched by NASA (CRS):
```python
%sql SELECT sum(payload_mass__kg_) AS total_payload_mass FROM SPACEXTBL WHERE customer = 'NASA (CRS)';
```

<img width="156" alt="Screenshot 2025-01-31 at 7 41 29 PM" src="https://github.com/user-attachments/assets/bdff950f-9ea0-46d9-87bd-830696ffa2ce" />

💡 The total payload carries by boosters launched by NASA is 45, 596 kg. 

### Task 4
Display average payload mass carried by booster version F9 v1.1:
```python
%sql SELECT avg(payload_mass__kg_) AS avg_payload_mass FROM SPACEXTBL WHERE booster_version like '%F9 v1.1%';
```

<img width="160" alt="Screenshot 2025-01-31 at 7 42 12 PM" src="https://github.com/user-attachments/assets/b6f07fdb-1b62-40bf-a896-0b978945e418" />

💡 The average payload mass carried by verion F9 v1.1 boosters is 2, 534.7 kg

### Task 5
List the date when the first succesful landing outcome in ground pad was acheived:
```python
%sql SELECT min(date) AS first_successful_landing FROM SPACEXTBL WHERE Landing_Outcome = 'Success (ground pad)';
```

<img width="182" alt="Screenshot 2025-01-31 at 7 42 46 PM" src="https://github.com/user-attachments/assets/e34fbcb9-6fd8-47b8-a7cf-562e0e6140d0" />

💡 The first successful landing outcome in ground pad was achieved on December 22, 2015.

### Task 6
List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000:
```python
%sql SELECT booster_version FROM SPACEXTBL WHERE Landing_Outcome = 'Success (drone ship)' AND payload_mass__kg_ between 4000 and 6000;
```

<img width="127" alt="Screenshot 2025-01-31 at 7 44 02 PM" src="https://github.com/user-attachments/assets/c207386f-b380-4056-b74b-3973fb94fa7e" />

💡 The 4 boosters which have success in drone ship with payload mass greater than 4000 but less than 6000 are listed above.

### Task 7
List the total number of successful and failure mission outcomes:
```python
%sql SELECT Mission_Outcome, count(*) AS total FROM SPACEXTBL GROUP BY Mission_Outcome;
```

<img width="270" alt="Screenshot 2025-01-31 at 7 44 37 PM" src="https://github.com/user-attachments/assets/e93cfef3-d431-4560-a554-b23f66cd33d2" />

### Task 8
List the names of the booster_versions which have carried the maximum payload mass. Use a subquery:
```python
%sql select booster_version from SPACEXTBL where payload_mass__kg_ = (select max(payload_mass__kg_) from SPACEXTBL);
```
<img width="127" alt="Screenshot 2025-01-31 at 7 45 20 PM" src="https://github.com/user-attachments/assets/248189bf-fb85-405c-a95b-8b7740cca153" />

💡 The booster versions which have carried the maximum payload mass are listed above.

### Task 9
List the records which will display the month names, failure landing_outcomes in drone ship ,booster versions, launch_site for the months in year 2015:
❗SQLLite does not support monthnames. So you need to use substr(Date, 6,2) as month to get the months and substr(Date,0,5)='2015' for year.
```python
%%sql select substr(Date, 6,2) as month, date, booster_version, launch_site, Landing_Outcome from SPACEXTBL
      where Landing_Outcome = 'Failure (drone ship)' and substr(Date,0,5)='2015';
```
<img width="497" alt="Screenshot 2025-01-31 at 7 46 11 PM" src="https://github.com/user-attachments/assets/7121147a-54d1-4e8a-ae9d-d08794a5aa34" />

💡 Two failed landing outcomes with a drone ship in 2015, one in January and April both from CCAFS LC-40.

### Task 10
Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order:
```python
%%sql select Landing_Outcome, count(*) as outcomes_rank from SPACEXTBL
      where date between '2010-06-04' and '2017-03-20'
      group by Landing_Outcome
      order by outcomes_rank desc;
```

<img width="279" alt="Screenshot 2025-01-31 at 7 46 52 PM" src="https://github.com/user-attachments/assets/d45218b2-b123-4a05-a886-d4f5fe4452a2" />

💡 Most common landing outcome is "No attempt."

***









