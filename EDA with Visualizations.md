# 🚀 SpaceX-AppliedDS
## Exploratory Data Analysis with Visualizations

We will predict if the Falcon 9 first stage will land successfully. SpaceX advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars; other providers cost upward of 165 million dollars each, much of the savings is due to the fact that SpaceX can reuse the first stage.

## ❗Objectives
- Perform Exploratory Data Analysis and Feature Engineering with `Pandas` and `Matplotlib`
- Preparing Data Feature Engineering

## Import Libraries
```python
import piplite
await piplite.install(['numpy'])
await piplite.install(['pandas'])
await piplite.install(['seaborn'])
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Exploratory Data Analysis
Let's read the SpaceX dataset into a Pandas dataframe and print the summary.
```python
from js import fetch
import io

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp = await fetch(URL)
dataset_part_2_csv = io.BytesIO((await resp.arrayBuffer()).to_py())
df=pd.read_csv(dataset_part_2_csv)
df.head(5)
```

<img width="896" alt="Screenshot 2025-01-31 at 9 21 58 PM" src="https://github.com/user-attachments/assets/d994b57c-fb97-4080-ad5e-18049ba1ea86" />


### TASK 1: Visualize the Relationship Between Flight Number and Launch Site
```python
sns.catplot(x='FlightNumber', y='LaunchSite', hue='Class', data=df, aspect=5)
plt.xlabel('Flight Number', fontsize=20)
plt.ylabel('Launch Site', fontsize=20)
plt.show()
```
1 = successful landing

0 = unsuccessful landing

<img width="927" alt="Screenshot 2025-01-31 at 9 22 57 PM" src="https://github.com/user-attachments/assets/ace65733-4116-4197-a9bc-963875793bcb" />

- Success rate varies for each launch.
- Success rate for Falcon 9 first stage landings deem to become prevelant as flight number increases.

### TASK 2: Visualize the Relationship Between Payload Mass and Launch Site
```python
sns.catplot(x='PayloadMass', y='LaunchSite', hue='Class', data=df, aspect = 5)
plt.xlabel('Payload Mass (kg)',fontsize=20)
plt.ylabel('Launch Site',fontsize=20)
plt.show()
```

<img width="919" alt="Screenshot 2025-01-31 at 9 27 48 PM" src="https://github.com/user-attachments/assets/2df8237d-cd54-4535-a859-b5a2f9e4ea16" />

- No strong correlation of the variables for site CCAFS SLC 40.
- Failed landings for site KSC LC 34A deem to be prevelant around a certain mass around, around 6000 kg.

### TASK 3: Visualize the Relationship Between Success Rate of Each Orbit Type
```python
sns.catplot(x= 'Orbit', y = 'Class', data = df.groupby('Orbit')['Class'].mean().reset_index(), kind = 'bar')
plt.xlabel('Orbit Type',fontsize=20)
plt.ylabel('Success Rate',fontsize=20)
plt.show()
```

<img width="525" alt="Screenshot 2025-01-31 at 9 31 26 PM" src="https://github.com/user-attachments/assets/32691205-f3da-4b7e-b558-5cdfb6eda0ba" />

- Orbits with 100% success rate are:
ES-L1,
GEO,
HEO,
SSO,

- Orbits with 0% success rate are:
SO

- Orbits with success rate between 50% and 85%:
GTO,
ISS,
LEO,
MEO,
PO,


### TASK 4: Visualize the Relationship Between FlightNumber and Orbit Type
```python
sns.catplot(x = 'FlightNumber', y = 'Orbit', hue = 'Class', data = df, aspect = 5)
plt.xlabel('Flight Number', fontsize = 20)
plt.ylabel('Orbit', fontsize = 20)
plt.show()
```

<img width="924" alt="Screenshot 2025-01-31 at 9 33 59 PM" src="https://github.com/user-attachments/assets/564a7c7b-549b-4c41-a0f6-212f9ed4f99d" />

- Positive correlation with flight number and success rate. Ie. Larger flight numbers deem to have higher success rates for all orbit types.

### TASK 5: Visualize the Relationship Between Payload Mass and Orbit type
```python
sns.catplot(x = 'PayloadMass', y = 'Orbit', hue = 'Class', data = df, aspect = 5)
plt.xlabel('Payload Mass (kg)', fontsize = 20)
plt.ylabel('Orbit', fontsize = 20)
plt.show()
```

<img width="909" alt="Screenshot 2025-01-31 at 9 37 17 PM" src="https://github.com/user-attachments/assets/910acd18-9010-4a2a-a751-bc68114e6a39" />

- No strong correlation between orbit type and payload mass.
- Some orbit types have higher success rates than others.

## Feature Engineering 
By now, we should have obtained some preliminary insights about how each important variable would affect the success rate, we will select the features that will be used in success prediction in the future,

```python
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()
```

<img width="882" alt="Screenshot 2025-01-31 at 9 39 43 PM" src="https://github.com/user-attachments/assets/cd09e7f7-c80b-4c2a-bd4e-7fb099650b48" />

### TASK 7: Create Dummy Variables to Categorical Columns
Use the function `get_dummies` and `features` dataframe to apply OneHotEncoder to the column Orbits, LaunchSite, LandingPad, and Serial. Assign the value to the variable features_one_hot, display the results using the method head. Your result dataframe must include all features including the encoded ones.
```python
features_one_hot = pd.get_dummies(features[['Orbit', 'LaunchSite', 'LandingPad', 'Serial']])
features_one_hot.head()
```

<img width="906" alt="Screenshot 2025-01-31 at 9 42 37 PM" src="https://github.com/user-attachments/assets/bf972f6a-1681-436f-81d1-8c2efb26f8b9" />

### TASK 8: Cast All Numeric Columns to `float64`
```python
features_one_hot.astype('float64')
```

<img width="910" alt="Screenshot 2025-01-31 at 9 44 09 PM" src="https://github.com/user-attachments/assets/94c6b281-a3db-48b5-9cbb-1ca042c9f9e4" />

***


