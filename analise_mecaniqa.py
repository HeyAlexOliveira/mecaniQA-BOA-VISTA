import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_excel("mecaniqa_dataset.xlsx")

df["Data"] = pd.to_datetime(df["Data"])
df = df.set_index("Data")
df = df.sort_index()

coluna_valores = df.columns[0]

decomposicao = seasonal_decompose(
    df[coluna_valores], model="additive", period=7
)

fig = decomposicao.plot()
fig.set_size_inches(12, 8)
fig.suptitle(
    "Decomposição da Série Temporal - MecâniQA",
    fontsize=14,
    fontweight="bold",
)

plt.tight_layout()
plt.show()
