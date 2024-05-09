import pandas as pd
import matplotlib.pyplot as plt

# Daten einlesen
df = pd.read_csv("Fokus/fokus_faktor.csv")
df.rename(columns={"lensposition": "Position des Fokus [Dioptrinen]", "faktor_x": "Faktor in X-Richtung",
          "faktor_y": "Faktor in Y-Richtung", "faktor_soll": "Faktor nach Näherungsformel"}, inplace=True)

df.set_index("Position des Fokus [Dioptrinen]", inplace=True)

# Daten anzeigen
print(df)

# Daten visualisieren
df.plot()

# Diagramm anzeigen
# plt.show()

# Diagramm speichern
plt.savefig("../Thesis/img/fokus_faktor_diagramm.pdf", format="pdf")
