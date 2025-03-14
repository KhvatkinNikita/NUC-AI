import pandas as pd

def data_loader():
    df = pd.read_csv('../data/dataset-ISTAC-C00022A_000005-~latest-observations.tsv', sep='\t', low_memory = False)

    islas = ['Tenerife', 'Gran Canaria', 'Lanzarote', 'Fuerteventura', 'La Palma', 'La Gomera', 'El Hierro'] # 7 main islands
    col = ['TERRITORIO', 'TIME_PERIOD_CODE', 'OBS_VALUE', 'FLUJO_ENERGIA'] # important columns

    _ = df[df['TERRITORIO'].isin(islas)][col] # take only the 7 main islands
    values_islas_total = _[_['FLUJO_ENERGIA'] == 'Total'].drop('FLUJO_ENERGIA', axis=1).reset_index(drop=True) # take total energy 
    values_islas_total['TIME_PERIOD_CODE'] = values_islas_total['TIME_PERIOD_CODE'].astype('datetime64[ns]')

    FIRST = values_islas_total['TIME_PERIOD_CODE'].min()
    LAST = values_islas_total['TIME_PERIOD_CODE'].max()
    print(f"Fecha inicio: {FIRST}\nFecha fin: {LAST}\nTotal number of days: {(LAST-FIRST)}")
    
    islas = ['Tenerife', 'Gran Canaria', 'Lanzarote', 'Fuerteventura', 'La Palma', 'La Gomera', 'El Hierro']

    isla_dfs = {}

    for isla in islas:
        isla_dfs[isla] = (
            values_islas_total[values_islas_total['TERRITORIO'] == isla]
            .drop('TERRITORIO', axis=1)
            .sort_values('TIME_PERIOD_CODE')
            .reset_index(drop=True)
        )
    return isla_dfs