import pandas as pd

df = pd.read_excel("mecaniqa_dataset.xlsx")

df["Data"] = pd.to_datetime(df["Data"])
df = df.set_index("Data")
df = df.sort_index()

print("Primeiros registros:")
print(df.head())

print("\nInformações do DataFrame:")
df.info()

print("\nTamanho da base:")
print(df.shape)