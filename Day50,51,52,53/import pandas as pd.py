import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Sample Dataset Creation ---
np.random.seed(42)
dates = pd.date_range(start="2025-01-01", periods=30, freq="D")
stores = ['Store_A', 'Store_B', 'Store_C']
categories = ['Electronics', 'Groceries', 'Clothing']

data = {
    'Date': np.tile(dates, len(stores)),
    'Store': np.repeat(stores, len(dates)),
    'Product_Category': np.random.choice(categories, len(dates) * len(stores)),
    'Sales_Amount': np.random.randint(100, 2000, len(dates) * len(stores)),
    'Quantity_Sold': np.random.randint(1, 20, len(dates) * len(stores))
}

df = pd.DataFrame(data)

# --- 2. MultiIndexing: Store + Date ---
df.set_index(['Store', 'Date'], inplace=True)
df.sort_index(inplace=True)

print("\nMultiIndexed DataFrame:")
print(df.head())

# --- 3. Reshape with melt, stack, unstack ---
# Reset index to work with melt
df_reset = df.reset_index()
melted = df_reset.melt(id_vars=['Store', 'Date', 'Product_Category'],
                       value_vars=['Sales_Amount', 'Quantity_Sold'],
                       var_name='Metric', value_name='Value')

# Unstack Metric to see side-by-side
reshaped = melted.set_index(['Store', 'Date', 'Product_Category', 'Metric']).unstack('Metric')
print("\nReshaped Metrics (unstacked):")
print(reshaped.head())

# --- 4. Rolling 7-Day Moving Average of Sales per Store ---
# Total sales per Store-Date
sales_per_day = df.groupby(['Store', 'Date'])['Sales_Amount'].sum().reset_index()

# Set index for rolling calculation
sales_per_day.set_index(['Store', 'Date'], inplace=True)
rolling_avg = sales_per_day.groupby(level='Store')['Sales_Amount'].rolling(window=7, min_periods=1).mean().rename('7Day_Moving_Avg').reset_index()

# Merge for comparison
sales_trend = pd.merge(sales_per_day.reset_index(), rolling_avg, on=['Store', 'Date'])
print("\nRolling Average Sales:")
print(sales_trend.head())

# --- 5. Vectorized Optimization: Flag high sales days (Sales > 1500) ---
sales_trend['High_Sales_Day'] = (sales_trend['Sales_Amount'] > 1500).astype(int)

# --- 6. .apply() instead of .iterrows() ---
# Add revenue per item sold using apply
df_reset['Revenue_per_Item'] = df_reset[['Sales_Amount', 'Quantity_Sold']].apply(
    lambda x: round(x.Sales_Amount / x.Quantity_Sold, 2) if x.Quantity_Sold else 0, axis=1)

print("\nSample with Revenue per Item:")
print(df_reset[['Store', 'Date', 'Sales_Amount', 'Quantity_Sold', 'Revenue_per_Item']].head())

# --- 7. Group by Product Category: Total & Average Sales ---
category_stats = df_reset.groupby('Product_Category').agg(
    Total_Revenue=('Sales_Amount', 'sum'),
    Avg_Sales=('Sales_Amount', 'mean'),
    Total_Quantity=('Quantity_Sold', 'sum')
).reset_index()

print("\nSales by Product Category:")
print(category_stats)

# --- 8. Visualizations using Pandas & Matplotlib ---

# Sales Trend per Store
plt.figure(figsize=(10, 6))
for store in sales_trend['Store'].unique():
    store_data = sales_trend[sales_trend['Store'] == store]
    plt.plot(store_data['Date'], store_data['Sales_Amount'], label=f'{store} Sales')
    plt.plot(store_data['Date'], store_data['7Day_Moving_Avg'], linestyle='--', label=f'{store} 7-Day Avg')

plt.title('Daily Sales and 7-Day Moving Average by Store')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Plot: Total Revenue per Product Category
plt.figure(figsize=(8, 5))
plt.bar(category_stats['Product_Category'], category_stats['Total_Revenue'])
plt.title('Total Revenue by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Revenue')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
