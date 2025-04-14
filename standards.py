# prediction_app/standards.py
NEMA_STANDARDS = {
    'Effluent_COD': 50,        # mg/L
    'Effluent_pH': (6, 9),     # pH range
    'Effluent_TSS': 30,        # mg/L
    'Effluent_TDS': 500,       # mg/L
    'Effluent_Conductivity': 1000,  # μS/cm
    'Effluent_Turbidity': 5,   # NTU
    'Effluent_Temperature': 40 # °C (max allowable increase)
}