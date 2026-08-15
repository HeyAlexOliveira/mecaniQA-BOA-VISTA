import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_excel("mecaniqa_dataset.xlsx")

df["Data"] = pd.to_datetime(df["Data"])
df = df.set_index("Data")
df = df.sort_index()

df["Trocas_Oleo"] = df["Trocas_Oleo"].interpolate()

resultado = seasonal_decompose(
    df["Trocas_Oleo"],
    model="additive",
    period=7
)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    4, 1,
    figsize=(12, 10),
    sharex=True
)

resultado.observed.plot(
    ax=ax1,
    title="Série Observada - Trocas de Óleo"
)

resultado.trend.plot(
    ax=ax2,
    title="Tendência"
)

resultado.seasonal.plot(
    ax=ax3,
    title="Sazonalidade"
)

resultado.resid.plot(
    ax=ax4,
    title="Ruído / Resíduos"
)

plt.suptitle(
    "Decomposição da Série Temporal - MecâniQA",
    fontsize=14
)

plt.tight_layout()
plt.show()
