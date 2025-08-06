
import pandas as pd


import numpy as np


df = pd.read_csv("covid_data.csv")  # Replace with your actual file


print("Basic Info:")
print(df.info())
print("\nSummary:")
print(df.describe())
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df.fillna(0, inplace=True)


top10_cases = df.sort_values(by="TotalCases", ascending=False).head(10)
print("\nTop 10 countries by total cases:")
print(top10_cases[["Country", "TotalCases"]])


df["ActiveCases"] = df["TotalCases"] - df["TotalRecovered"] - df["TotalDeaths"]
print("\nSample Active Cases:")
print(df[["Country", "ActiveCases"]].head())


df["DeathRate"] = (df["TotalDeaths"] / df["TotalCases"]) * 100
highest_death_rate = df[df["TotalCases"] > 0].sort_values(by="DeathRate", ascending=False).head(1)
print("\nCountry with highest death rate:")
print(highest_death_rate[["Country", "DeathRate"]])


if "Continent" in df.columns:
    continent_cases = df.groupby("Continent")["TotalCases"].sum().reset_index()
    print("\nTotal cases by continent:")
    print(continent_cases)
else:
    print("\nNo 'Continent' column found.")


df["RecoveryRate"] = (df["TotalRecovered"] / df["TotalCases"]) * 100
average_recovery_rate = df["RecoveryRate"].mean()
print(f"\nAverage recovery rate: {average_recovery_rate:.2f}%")

high_recovery = df[df["RecoveryRate"] > 90]
print("\nCountries with recovery rate above 90%:")
print(high_recovery[["Country", "RecoveryRate"]])

df_sorted_new_cases = df.sort_values(by="NewCases", ascending=False)
print("\nTop 5 countries by new cases today:")
print(df_sorted_new_cases[["Country", "NewCases"]].head())

df.to_csv("cleaned_covid_data.csv", index=False)
print("\nCleaned data saved to cleaned_covid_data.csv")


plt.figure(figsize=(12, 6))
sns.barplot(x="TotalCases", y="Country", data=top10_cases, palette="Reds_r")
plt.title("Top 10 Countries by Total COVID-19 Cases")
plt.xlabel("Total Cases")
plt.ylabel("Country")
plt.tight_layout()
plt.show()


df_reg = df[df["NewCases"] > 0]
X = df_reg[["NewCases"]]
y = df_reg["TotalCases"]

model = LinearRegression()
model.fit(X, y)
future_cases = model.predict([[10000]])  # Predict when 10,000 new cases
print(f"\nPredicted total cases for 10,000 new cases: {int(future_cases[0])}")
