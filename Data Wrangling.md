# 🚀 SpaceX-AppliedDS
## Data Wrangling/Cleaning
The dataset being used is a IBM copy of a reponse from a public API with launch data in JSON format, and web data from wikipedia collected through webscraping, along with other datasets from the course.

When we analyze the dataset we can see there are rows with missing values
```python
data_falcon9.isnull().sum()
```
<img width="191" alt="Screenshot 2025-01-31 at 7 12 15 PM" src="https://github.com/user-attachments/assets/ef2ba6ca-61c1-4aed-a9e9-b71eb6e1b8e7" />

❗`LandingPad` will retain None values to represent when landing pads were not used.

Calculate the mean value of PayloadMass column

```python
payloadmass_mean = data_falcon9['PayloadMass'].mean()
```
Replace the np.nan values with its mean value
```python
data_falcon9['PayloadMass'].replace(np.nan, payload_mass_mean, inplace = True)
```
The number of missing values in `PayloadMass` should now be changed to zero.
```python
data_falcon9.isnull().sum()
```
<img width="174" alt="Screenshot 2025-01-31 at 7 23 44 PM" src="https://github.com/user-attachments/assets/574b62ae-4d5b-4860-b99a-6fdc96d9a09b" />

