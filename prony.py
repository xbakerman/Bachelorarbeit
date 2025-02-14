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
conv = pv.load.conventions(['E'])

# Daten einlesen
# Testdatei aus dem repository mit Speicher und Verlustmodul
data = pv.load.file('./Examples/freq_user_raw.csv')
df_raw, arr_RefT, units = pv.load.user_raw(data, 'freq', 'E')

# Einheiten Check
check = pv.load.check_units(units, modul='E')
print("Einheitenprüfung erfolgreich:", check is None)

#---------------------------------------------------------#

# Bestimme Referenztemperatur basierend auf dem Maximum von E_loss (Glas)
Tg_index = df_raw['E_loss'].idxmax()
RefT_initial = df_raw.loc[Tg_index, 'T']  # Glasübergangstemperatur als Referenz

# Anpassung der Referenztemperatur basierend auf vorhandenen Temperaturen
arr_RefT = df_raw['T_round'].dropna().unique()
arr_RefT = pd.Series(np.sort(arr_RefT))

if RefT_initial in arr_RefT.values:
    RefT = RefT_initial
else:
    RefT = arr_RefT.iloc[(arr_RefT - RefT_initial).abs().argsort()[:1]].values[0]

print(f'Empfohlene Referenztemperatur (Tg) basierend auf E_loss-Maximum: {RefT} °C')


#---------------------------------------------------------#
# Shift-Faktoren berechnen

df_aT, df_shift = pv.master.get_aT(df_raw, RefT)


#---------------------------------------------------------#
# Masterkurve erstellen

df_master = pv.master.get_curve(df_raw=df_raw, df_aT=df_aT, RefT=RefT)
fig_master_shifted, figure_master_shift_lax = pv.master.plot_shift(df_raw=df_raw, df_master=df_master, units=units)
plt.title("Masterkurve")
plt.show()


#---------------------------------------------------------#
# Shiftfunktionen erhalten 

df_WLF = pv.shift.fit_WLF(RefT=RefT, df_aT=df_aT)
df_poly_C, df_poly_K = pv.shift.fit_poly(df_aT=df_aT)
figure_shift, df_shift = pv.shift.plot(df_aT=df_aT, df_WLF=df_WLF, df_C=df_poly_C)
plt.title("Shiftfunktionen")
plt.show()


#---------------------------------------------------------#
# Glätten der Masterkurve

gl_faktor = 4
df_master_smooth = pv.master.smooth(df_master, gl_faktor)
figure_smoothed = pv.master.plot_smooth(df_master_smooth, units)
plt.title("Geglättete Masterkurve")
plt.show()


#---------------------------------------------------------#
# Diskretisierung der Anzahl der Pronyparameter

df_discr = pv.prony.discretize(df_master_smooth, nprony=6)
figure_discr = pv.prony.plot_dis(df_master_smooth, df_discr, units)
plt.title("Diskretisierung der Pronyparameter")
plt.show()


#---------------------------------------------------------#
# Fitten der Pronyparameter

prony_param, df_GenMaxwell = pv.prony.fit(df_discr, df_master_smooth, opt=True)
figure_fit = pv.prony.plot_fit(df_master_smooth, df_GenMaxwell, units)
plt.title("Fitten der Pronyparameter")
plt.show()
print("Pronyparameter:", prony_param)
Maxwellparam = pv.out.to_csv(df_GenMaxwell, units=units, filepath='Maxwell_Model.csv')
# Speichern der df_terms in eine CSV-Datei
df_terms = prony_param['df_terms']
#df_terms.to_csv('prony_parameter.csv', index=False, sep=';', decimal=',', index_label='i')
pv.out.to_csv(df_terms, units=units, filepath='prony_parameter.csv', index_label='i')


#---------------------------------------------------------#
# Plotten des generalzied Maxwell Modells

figure_GenMaxwell = pv.prony.plot_GMaxw(df_GenMaxwell, units)
plt.title("Generalized Maxwell Modell")
plt.show()


