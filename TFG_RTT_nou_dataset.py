"""
TFG - Dataset nou UVIndoorLoc-RTT&RSSI
Compensació d'offset + k-NN posicionament
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# ─────────────────────────────────────────────────────────────
# 1. CÀRREGA DE DADES
# ─────────────────────────────────────────────────────────────
df_aps  = pd.read_csv('00_aps.csv')
df_loc  = pd.read_csv('00_locations.csv')
df_train = pd.read_csv('01_training.csv')
df_test1 = pd.read_csv('02_test1.csv')
df_test2 = pd.read_csv('02_test2.csv')

print(f"Training: {len(df_train)} mostres")
print(f"Test1 (mateix dia): {len(df_test1)} mostres")
print(f"Test2 (dia diferent): {len(df_test2)} mostres")

# Comprovar unitats: les distàncies estan en mm
print(f"\nExemple dist_ap1 (primeres 5 files): {df_train['dist_ap1'].head().tolist()}")

# Les distàncies estan en mm → convertim a metres
MISSING_MM = -1000000
AP_IDS = list(range(1, 21))

def mm_to_m(df):
    """Converteix columnes dist_ap i std_ap de mm a metres."""
    df = df.copy()
    for ap_id in AP_IDS:
        col = f'dist_ap{ap_id}'
        std_col = f'std_ap{ap_id}'
        if col in df.columns:
            mask = df[col] != MISSING_MM
            df.loc[mask, col] = df.loc[mask, col] / 1000.0
        if std_col in df.columns:
            mask_std = df[std_col] != 0
            df.loc[mask_std, std_col] = df.loc[mask_std, std_col] / 1000.0
    return df

MISSING = -1000.0  # nou valor missing en metres

df_train = mm_to_m(df_train)
df_test1 = mm_to_m(df_test1)
df_test2 = mm_to_m(df_test2)

# ─────────────────────────────────────────────────────────────
# 2. CALCULAR DISTÀNCIES REALS GEOMÈTRIQUES
# ─────────────────────────────────────────────────────────────
def calcular_d_real(df, df_aps):
    df = df.copy()
    for _, ap in df_aps.iterrows():
        ap_id = int(ap['id'])
        col_real = f'd_real_ap{ap_id}'
        dx = df['x'] - ap['x']
        dy = df['y'] - ap['y']
        dz = df['z'] - ap['z']
        df[col_real] = np.sqrt(dx**2 + dy**2 + dz**2)
    return df

df_train = calcular_d_real(df_train, df_aps)
df_test1 = calcular_d_real(df_test1, df_aps)
df_test2 = calcular_d_real(df_test2, df_aps)

# ─────────────────────────────────────────────────────────────
# 3. ESTADÍSTIQUES OFFSET SISTEMÀTIC
# ─────────────────────────────────────────────────────────────
print("\n=== OFFSET SISTEMÀTIC PER DISPOSITIU (TRAIN) ===")
for device in ['POCO', 'GP10', 'S24U']:
    df_dev = df_train[df_train['smartphone'] == device]
    errors_all = []
    for ap_id in AP_IDS:
        col_meas = f'dist_ap{ap_id}'
        col_real = f'd_real_ap{ap_id}'
        mask = df_dev[col_meas] != MISSING
        if mask.sum() > 0:
            errs = df_dev.loc[mask, col_meas] - df_dev.loc[mask, col_real]
            errors_all.append(errs)
    if errors_all:
        all_e = pd.concat(errors_all)
        print(f"{device}: ME={all_e.mean():.3f} m | RMSE={np.sqrt((all_e**2).mean()):.3f} m | N={len(all_e)}")

# ─────────────────────────────────────────────────────────────
# 4. TÈCNIQUES DE COMPENSACIÓ
# ─────────────────────────────────────────────────────────────

def apply_mean_offset(df_train, df_test, device):
    """Tècnica 1: Offset mitjà estimat al training."""
    df_dev_train = df_train[df_train['smartphone'] == device]
    df_out = df_test.copy()
    
    for ap_id in AP_IDS:
        col_meas  = f'dist_ap{ap_id}'
        col_real  = f'd_real_ap{ap_id}'
        col_corr  = f'corr_ap{ap_id}'
        
        mask_tr = df_dev_train[col_meas] != MISSING
        if mask_tr.sum() == 0:
            df_out[col_corr] = df_out[col_meas].astype(float)
            continue
        
        offset = (df_dev_train.loc[mask_tr, col_meas].values - 
                  df_dev_train.loc[mask_tr, col_real].values).mean()
        
        df_out[col_corr] = df_out[col_meas].astype(float)
        mask_te = df_out[col_meas] != MISSING
        df_out.loc[mask_te, col_corr] = df_out.loc[mask_te, col_meas] - offset
    
    return df_out


def apply_3sigma_offset(df_train, df_test, device):
    """Tècnica 2: Filtre 3σ sobre training + offset mitjà."""
    df_dev_train = df_train[df_train['smartphone'] == device]
    df_out = df_test.copy()
    
    for ap_id in AP_IDS:
        col_meas = f'dist_ap{ap_id}'
        col_real = f'd_real_ap{ap_id}'
        col_corr = f'corr_ap{ap_id}'
        
        mask_tr = df_dev_train[col_meas] != MISSING
        vals_tr  = df_dev_train.loc[mask_tr, col_meas].values
        reals_tr = df_dev_train.loc[mask_tr, col_real].values
        
        if len(vals_tr) == 0:
            df_out[col_corr] = df_out[col_meas].astype(float)
            continue
        
        mu, sig = vals_tr.mean(), vals_tr.std()
        mask_3s = (vals_tr >= mu - 3*sig) & (vals_tr <= mu + 3*sig)
        
        if mask_3s.sum() == 0:
            df_out[col_corr] = df_out[col_meas].astype(float)
            continue
        
        offset = (vals_tr[mask_3s] - reals_tr[mask_3s]).mean()
        
        df_out[col_corr] = df_out[col_meas].astype(float)
        mask_te = df_out[col_meas] != MISSING
        df_out.loc[mask_te, col_corr] = df_out.loc[mask_te, col_meas] - offset
    
    return df_out


# ─────────────────────────────────────────────────────────────
# 5. K-NN POSICIONAMENT
# ─────────────────────────────────────────────────────────────

def preparar_fingerprint(df, scenario, device, col_prefix='dist_ap'):
    """Prepara matriu X (fingerprints) i y (posicions) per a un escenari."""
    sc_str  = f'S0{scenario}'
    ap_ids  = df_aps[df_aps['scenario'] == scenario]['id'].tolist()
    cols    = [f'{col_prefix}{ap_id}' for ap_id in ap_ids]
    
    # Filtrar per escenari i dispositiu
    df_f = df[(df['scenario'] == sc_str) & 
              (df['smartphone'] == device)].copy()
    
    if len(df_f) == 0:
        return np.array([]), np.array([]), df_f
    
    # Comprovar que les columnes existeixen
    cols_exist = [c for c in cols if c in df_f.columns]
    if not cols_exist:
        return np.array([]), np.array([]), df_f
    
    # Substituir missing per NaN i imputar amb mediana
    X = df_f[cols_exist].copy().astype(float)
    X[X == MISSING] = np.nan
    for col in cols_exist:
        med = X[col].median()
        X[col] = X[col].fillna(med)
    
    y = df_f[['x', 'y']].values
    return X.values, y, df_f


def knn_positioning(X_train, y_train, X_test, k=3):
    dists   = cdist(X_test, X_train, metric='euclidean')
    idx_knn = np.argsort(dists, axis=1)[:, :k]
    y_pred  = np.mean(y_train[idx_knn], axis=1)
    return y_pred


def error_posicionament(y_true, y_pred):
    return np.sqrt(((y_true - y_pred)**2).sum(axis=1))


# ─────────────────────────────────────────────────────────────
# 6. EXPERIMENTS
# ─────────────────────────────────────────────────────────────
resultats = []

for device in ['POCO', 'GP10', 'S24U']:
    # Aplicar compensacions
    df_test1_off    = apply_mean_offset(df_train, df_test1, device)
    df_test2_off    = apply_mean_offset(df_train, df_test2, device)
    df_test1_3s_off = apply_3sigma_offset(df_train, df_test1, device)
    df_test2_3s_off = apply_3sigma_offset(df_train, df_test2, device)
    
    for scenario in [1, 2, 3]:
        sc_str = f'S0{scenario}'
        
        # Training fingerprints (sense compensació)
        X_tr, y_tr, _ = preparar_fingerprint(df_train, scenario, device, 'dist_ap')
        if len(X_tr) == 0:
            continue
        
        for test_label, df_te, df_te_off, df_te_3s_off in [
            ('Test1 (mateix dia)',    df_test1, df_test1_off, df_test1_3s_off),
            ('Test2 (dia diferent)',  df_test2, df_test2_off, df_test2_3s_off),
        ]:
            def afegir(tecnica, X_te, y_te):
                if len(X_te) == 0: return
                y_pred = knn_positioning(X_tr, y_tr, X_te, k=3)
                errs   = error_posicionament(y_te, y_pred)
                resultats.append({
                    'Dispositiu': device, 'Escenari': sc_str,
                    'Test': test_label, 'Tècnica': tecnica,
                    'N': len(errs),
                    'MAE':  round(errs.mean(), 3),
                    'RMSE': round(np.sqrt((errs**2).mean()), 3)
                })
            
            X_te, y_te, _           = preparar_fingerprint(df_te,         scenario, device, 'dist_ap')
            X_te_off, y_te_off, _   = preparar_fingerprint(df_te_off,     scenario, device, 'corr_ap')
            X_te_3s, y_te_3s, _     = preparar_fingerprint(df_te_3s_off,  scenario, device, 'corr_ap')
            
            afegir('Sense compensació', X_te,     y_te)
            afegir('Offset mitjà',      X_te_off, y_te_off)
            afegir('3σ + Offset',       X_te_3s,  y_te_3s)

df_res = pd.DataFrame(resultats)
print("\n=== RESULTATS k-NN (MAE en metres) ===")
print(df_res.to_string())

# ─────────────────────────────────────────────────────────────
# 7. GRÀFICA COMPARATIVA
# ─────────────────────────────────────────────────────────────
devices = ['POCO', 'GP10', 'S24U']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Error de posicionament k-NN per tècnica de compensació\n(Test1 - mateix dia)", fontsize=12)

colors = {'Sense compensació': '#e74c3c', 'Offset mitjà': '#3498db', '3σ + Offset': '#2ecc71'}

for ax, device in zip(axes, devices):
    df_dev = df_res[(df_res['Dispositiu'] == device) & 
                    (df_res['Test'] == 'Test1 (mateix dia)')]
    
    tecniques = list(colors.keys())
    escenaris = ['S01', 'S02', 'S03']
    x = np.arange(len(escenaris))
    width = 0.25
    
    for i, tec in enumerate(tecniques):
        vals = []
        for sc in escenaris:
            row = df_dev[(df_dev['Escenari']==sc) & (df_dev['Tècnica']==tec)]
            vals.append(row['MAE'].values[0] if len(row) > 0 else 0)
        ax.bar(x + i*width, vals, width, label=tec, color=colors[tec], alpha=0.85)
    
    ax.set_title(device)
    ax.set_xlabel('Escenari')
    ax.set_ylabel('MAE posicionament (m)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(escenaris)
    ax.legend(fontsize=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

plt.tight_layout()
plt.savefig('knn_posicionament.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGràfica guardada com 'knn_posicionament.png'")
