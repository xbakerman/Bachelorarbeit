import pyvisco as pv
from pyvisco import styles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyvisco import master
import matplotlib

# Setze das Backend für Matplotlib
matplotlib.use('TkAgg')

# Konvention erstellen je nach Testdaten
conv = pv.load.conventions(['G'])

 

# Daten einlesen
# Testdatei aus dem Artikel https://www.researchgate.net/publication/341077277_Prony_series_calculation_for_viscoelastic_behavior_modeling_of_structural_adhesives_from_DMA_data
data = pv.load.file('./Examples/DMA_data.csv')
RefT = 30
df_master, units = pv.load.user_master(data, 'freq', RefT, 'G')
print(df_master)
print(units)


#---------------------------------------------------------#
# Subplots erstellen
fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Speicher-Modul plotten
axs[0].plot(df_master["f"], df_master["G_stor"], marker="o", linestyle="-", color="black", label="Storage Modulus")
axs[0].set_ylabel("Storage Modulus (GPa)")
axs[0].set_title("Storage Modulus vs. Frequency")
axs[0].grid(True)
axs[0].legend()

# Verlust-Modul plotten
axs[1].plot(df_master["f"], df_master["G_loss"], marker="s", linestyle="-", color="red", label="Loss Modulus")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Loss Modulus (GPa)")
axs[1].set_title("Loss Modulus vs. Frequency")
axs[1].grid(True)
axs[1].legend()



#Plotten der Masterkurve logarithmisch skaliert
pv.master.plot(df_master, units)
plt.title("Masterkurve logarithmisch")
plt.show()

#---------------------------------------------------------#
gl_faktor = 5
df_master_smooth = pv.master.smooth(df_master, gl_faktor)
figure_smoothed = pv.master.plot_smooth(df_master_smooth, units)
plt.title("Geglättete Masterkurve")
plt.show()


#---------------------------------------------------------#
# Diskretisierung der Anzahl der Pronyparameter

df_discr = pv.prony.discretize(df_master_smooth, window='exact', nprony=8)
figure_discr = pv.prony.plot_dis(df_master_smooth, df_discr, units)
plt.title("Diskretisierung der Pronyparameter")
plt.show()


#---------------------------------------------------------#
# Fitten der Pronyparameter

prony_param, df_GenMaxwell = pv.prony.fit(df_discr, df_master_smooth)
figure_fit = pv.prony.plot_fit(df_master_smooth, df_GenMaxwell, units)
plt.title("Fitten der Pronyparameter")
plt.show()
print("Pronyparameter:", prony_param)
Maxwellparam = pv.out.to_csv(df_GenMaxwell, units=units, filepath='Maxwell_Model_master.csv')
# Speichern der df_terms in eine CSV-Datei
df_terms = prony_param['df_terms']
#df_terms.to_csv('prony_parameter.csv', index=False, sep=';', decimal=',', index_label='i')
pv.out.to_csv(df_terms, units=units, filepath='prony_parameter_master.csv', index_label='i')


#---------------------------------------------------------#
# Plotten des generalzied Maxwell Modells

figure_GenMaxwell = pv.prony.plot_GMaxw(df_GenMaxwell, units)
plt.title("Generalized Maxwell Modell")
plt.show()


