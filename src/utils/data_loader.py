import pandas as pd
import requests
import io
import os

def local_data_loader(verbose = True):
    """
    Manual data loader of the ISTAC dataset (2020-2024)
    """
    df = pd.read_csv('../data/dataset-ISTAC-C00022A_000005-~latest-observations.tsv', sep='\t', low_memory = False)

    islas = ['Tenerife', 'Gran Canaria', 'Lanzarote', 'Fuerteventura', 'La Palma', 'La Gomera', 'El Hierro'] # 7 main islands
    col = ['TERRITORIO', 'TIME_PERIOD_CODE', 'OBS_VALUE', 'FLUJO_ENERGIA'] # important columns

    _ = df[df['TERRITORIO'].isin(islas)][col] # take only the 7 main islands
    values_islas_total = _[_['FLUJO_ENERGIA'] == 'Total'].drop('FLUJO_ENERGIA', axis=1).reset_index(drop=True) # take total energy 
    values_islas_total['TIME_PERIOD_CODE'] = values_islas_total['TIME_PERIOD_CODE'].astype('datetime64[ns]')

    FIRST = values_islas_total['TIME_PERIOD_CODE'].min()
    LAST = values_islas_total['TIME_PERIOD_CODE'].max()

    if verbose:
        print(f"Fecha inicio: {FIRST}\nFecha fin: {LAST}\nTotal number of days: {(LAST-FIRST)}")

    isla_dfs = {}

    for isla in islas:
        isla_dfs[isla] = (
            values_islas_total[values_islas_total['TERRITORIO'] == isla]
            .drop('TERRITORIO', axis=1)
            .sort_values('TIME_PERIOD_CODE')
            .reset_index(drop=True)
        )
    return isla_dfs

def online_data_loader(verbose = True):
    """
    Online data loader of the ISTAC dataset (2020-2024)
    """
    url = 'https://datos.canarias.es/api/estadisticas/statistical-resources/v1.0/datasets/ISTAC/C00022A_000005/~latest.csv'
    response = requests.get(url) # HTTP GET request
    
    if response.status_code != 200:
        raise RuntimeError(f'Failed to fetch data: HTTP {response.status_code}')
    
    df = pd.read_csv(io.StringIO(response.text), low_memory = False)

    islas = ['Tenerife', 'Gran Canaria', 'Lanzarote', 'Fuerteventura', 'La Palma', 'La Gomera', 'El Hierro'] # 7 main islands
    col = ['TERRITORIO#es', 'TIME_PERIOD_CODE', 'OBS_VALUE', 'FLUJO_ENERGIA#es'] # important columns

    _ = df[df['TERRITORIO#es'].isin(islas)][col] # take only the 7 main islands
    values_islas_total = _[_['FLUJO_ENERGIA#es'] == 'Total'].drop('FLUJO_ENERGIA#es', axis=1).reset_index(drop=True) # take total energy 
    values_islas_total['TIME_PERIOD_CODE'] = values_islas_total['TIME_PERIOD_CODE'].astype('datetime64[ns]')

    FIRST = values_islas_total['TIME_PERIOD_CODE'].min()
    LAST = values_islas_total['TIME_PERIOD_CODE'].max()
    if verbose:
        print(f"Fecha inicio: {FIRST}\nFecha fin: {LAST}\nTotal number of days: {(LAST-FIRST)}")

    isla_dfs = {}

    for isla in islas:
        isla_dfs[isla] = (
            values_islas_total[values_islas_total['TERRITORIO#es'] == isla]
            .drop('TERRITORIO#es', axis=1)
            .sort_values('TIME_PERIOD_CODE')
            .reset_index(drop=True)
        )
    return isla_dfs

def data_loader(verbose = True, local = True):
    if local:
        """
        Manual data loader of the ISTAC dataset (2020-2024)
        """
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data'))
        df = pd.read_csv(os.path.join(data_path, 'dataset-ISTAC-C00022A_000005-~latest-observations.tsv'), sep='\t', low_memory = False)

        islas = ['Tenerife', 'Gran Canaria', 'Lanzarote', 'Fuerteventura', 'La Palma', 'La Gomera', 'El Hierro'] # 7 main islands
        col = ['TERRITORIO', 'TIME_PERIOD_CODE', 'OBS_VALUE', 'FLUJO_ENERGIA'] # important columns

        _ = df[df['TERRITORIO'].isin(islas)][col] # take only the 7 main islands
        values_islas_total = _[_['FLUJO_ENERGIA'] == 'Total'].drop('FLUJO_ENERGIA', axis=1).reset_index(drop=True) # take total energy 
        values_islas_total['TIME_PERIOD_CODE'] = values_islas_total['TIME_PERIOD_CODE'].astype('datetime64[ns]')

        FIRST = values_islas_total['TIME_PERIOD_CODE'].min()
        LAST = values_islas_total['TIME_PERIOD_CODE'].max()

        if verbose:
            print(f"Fecha inicio: {FIRST}\nFecha fin: {LAST}\nTotal number of days: {(LAST-FIRST)}")

        isla_dfs = {}

        for isla in islas:
            isla_dfs[isla] = (
                values_islas_total[values_islas_total['TERRITORIO'] == isla]
                .drop('TERRITORIO', axis=1)
                .sort_values('TIME_PERIOD_CODE')
                .reset_index(drop=True)
            )
        return isla_dfs
    else: 
        """
        Online data loader of the ISTAC dataset (2020-2024)
        """
        url = 'https://datos.canarias.es/api/estadisticas/statistical-resources/v1.0/datasets/ISTAC/C00022A_000005/~latest.csv'
        response = requests.get(url) # HTTP GET request

        if response.status_code != 200:
            raise RuntimeError(f'Failed to fetch data: HTTP {response.status_code}')

        df = pd.read_csv(io.StringIO(response.text), low_memory = False)

        islas = ['Tenerife', 'Gran Canaria', 'Lanzarote', 'Fuerteventura', 'La Palma', 'La Gomera', 'El Hierro'] # 7 main islands
        col = ['TERRITORIO#es', 'TIME_PERIOD_CODE', 'OBS_VALUE', 'FLUJO_ENERGIA#es'] # important columns

        _ = df[df['TERRITORIO#es'].isin(islas)][col] # take only the 7 main islands
        values_islas_total = _[_['FLUJO_ENERGIA#es'] == 'Total'].drop('FLUJO_ENERGIA#es', axis=1).reset_index(drop=True) # take total energy 
        values_islas_total['TIME_PERIOD_CODE'] = values_islas_total['TIME_PERIOD_CODE'].astype('datetime64[ns]')

        FIRST = values_islas_total['TIME_PERIOD_CODE'].min()
        LAST = values_islas_total['TIME_PERIOD_CODE'].max()
        if verbose:
            print(f"Fecha inicio: {FIRST}\nFecha fin: {LAST}\nTotal number of days: {(LAST-FIRST)}")

        isla_dfs = {}

        for isla in islas:
            isla_dfs[isla] = (
                values_islas_total[values_islas_total['TERRITORIO#es'] == isla]
                .drop('TERRITORIO#es', axis=1)
                .sort_values('TIME_PERIOD_CODE')
                .reset_index(drop=True)
            )
        return isla_dfs