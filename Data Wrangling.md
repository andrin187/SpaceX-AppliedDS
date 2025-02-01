# 🚀 SpaceX-AppliedDS
## Data Wrangling/Cleaning
The dataset being used is a IBM copy of a reponse from a public API with launch data in JSON format, and web data from wikipedia collected through webscraping, along with other datasets from the course.

When we analyze the dataset we can see there are rows with missing values
```python
data_falcon9.isnull().sum()
```
<img width="191" alt="Screenshot 2025-01-31 at 7 12 15 PM" src="https://github.com/user-attachments/assets/ef2ba6ca-61c1-4aed-a9e9-b71eb6e1b8e7" />
