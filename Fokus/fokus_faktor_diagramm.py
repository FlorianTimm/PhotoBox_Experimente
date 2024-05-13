import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# read xml file
import xml.etree.ElementTree as ET
tree = ET.parse('Fokus/project_12345_vorkalibriert_fxy123_cameras.xml')
root = tree.getroot()

# get all camera parameters
print(root.tag)
sensors = root.findall(".//sensor")
print(sensors)

liste = []
for sensor in sensors:
    sensor_parameters = sensor.findall("calibration")
    for sensor_parameter in sensor_parameters:
        if (sensor_parameter.attrib["class"] == "adjusted"):
            liste.append({
                "Fokus [dpt]": int(sensor.attrib["label"][-1]),
                "f": float(sensor_parameter.find("f").text),
                "cx": float(sensor_parameter.find("cx").text),
                "cy": float(sensor_parameter.find("cy").text),
                "k1": float(sensor_parameter.find("k1").text),
                "k2": float(sensor_parameter.find("k2").text),
                "k3": float(sensor_parameter.find("k3").text),
            })

ist = pd.DataFrame(liste)
print(ist)


def gerade(x, a, b):
    return a * x + b


fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, col in enumerate(ist.columns.values[1:]):
    m = i // 3
    n = i % 3
    # ist.plot(x="Fokus [dpt]",
    #         y=[col], kind="scatter", ax=axes[m][n], sharex=True)
    k_group = ist.groupby("Fokus [dpt]")

    axes[m][n].boxplot(k_group[col].apply(
        list), positions=list(k_group.groups.keys()), flierprops=dict(marker='o', markersize=4))

    popt, pcov = curve_fit(gerade, ist['Fokus [dpt]'], ist[col])
    print(popt)
    x = np.array(list(range(0, 7)))
    y = gerade(x, *popt)
    axes[m][n].plot(x, y, color="red", label=col + " = %4.2f + %4.2f * x" %
                    (popt[0], popt[1]))

    if col == "f":
        def naehrung(x):
            return 1/(1/0.00474-x)/0.0000014
        x = np.array(list(range(0, 7)))
        y = naehrung(x)
        axes[m][n].plot(x, y, color="green", label="gemäß Datenblatt")

    axes[m][n].legend()
    axes[m][n].set_xlabel("Fokus [dpt]")
    axes[m][n].set_ylabel(col, rotation=0, labelpad=-1, y=1)


# Diagramm speichern
fig.tight_layout()
plt.savefig(
    "../Thesis/img/naeherungswerte_diagramm.pdf", format="pdf")

plt.show()
