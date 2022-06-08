def select_lag_acf(time_series, max_lag):
    """ Seleciona os melhores lags usando ACF

        Args:
            time_series (Series): Série temporal.
            max_lag (int): Número máximo de lags.
        Returns:
            Sequência dos melhores LAGS.
    """
    from statsmodels.tsa.stattools import acf
    acf_x, confint = acf(time_series, nlags=max_lag, alpha=.05, fft=False)

    limiar_superior = confint[:, 1] - acf_x
    limiar_inferior = confint[:, 0] - acf_x
    lags_selecionados = []

    for i in range(1, max_lag + 1):

        if acf_x[i] >= limiar_superior[i] or acf_x[i] <= limiar_inferior[i]:
            lags_selecionados.append(i - 1)  # -1 por conta que o lag 1 em python é o 0

    if len(lags_selecionados) == 0:
        print('NENHUM LAG POR ACF')
        lags_selecionados = [i for i in range(max_lag)]

    lags_selecionados = [max_lag - (i + 1) for i in lags_selecionados]
    lags_selecionados = sorted(lags_selecionados, key=int)
    lags_selecionados.append(max_lag)

    return lags_selecionados


def select_lag_pacf(time_series, max_lag):
    """ Seleciona os melhores lags usando ACF

        Args:
            time_series (Series): Série temporal.
            max_lag (int): Número máximo de lags.
        Returns:
            Sequência dos melhores LAGS.
    """
    from statsmodels.tsa.stattools import pacf
    acf_x, confint = pacf(time_series, nlags=max_lag, alpha=.05)
    limiar_superior = confint[:, 1] - acf_x
    limiar_inferior = confint[:, 0] - acf_x
    lags_selecionados = []

    for i in range(1, max_lag + 1):

        if acf_x[i] >= limiar_superior[i] or acf_x[i] <= limiar_inferior[i]:
            lags_selecionados.append(i - 1)  # -1 por conta que o lag 1 em python é o 0

    if len(lags_selecionados) == 0:
        print('NENHUM LAG POR ACF')
        lags_selecionados = [i for i in range(max_lag)]

    lags_selecionados = [max_lag - (i + 1) for i in lags_selecionados]
    lags_selecionados = sorted(lags_selecionados, key=int)
    lags_selecionados.append(max_lag)

    return lags_selecionados


def create_windows(time_series, n_in=3, n_out=1, dropnan=True):
    """ Divide uma série temporal usando um algoritmo de janelas

        Args:
            time_series (Series): Série temporal.
            n_in (int): Número padrão de entrada.
            n_out (int): Número padrão de saída.
            dropnan (bool): Remover valores nulos.
        Returns:
            Serie temporal em janelas
    """
    from pandas import DataFrame, concat

    df_ts = DataFrame(time_series)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df_ts.shift(i))

    for i in range(0, n_out):
        cols.append(df_ts.shift(-i))

    agg = concat(cols, axis=1)

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg.values


def get_fixed_sample_sizes(max_window_size: int, serie: list, train_perc: float, vali_perc: float):
    if vali_perc == 0:
        return None, int(len(create_windows(serie, max_window_size)) * (1 - train_perc))
    else:
        fixed_vali_perc = int(len(create_windows(serie, max_window_size)) * vali_perc)
        fixed_test_perc = int(len(create_windows(serie, max_window_size)) * (1 - (train_perc + vali_perc)))

        return fixed_vali_perc, fixed_test_perc


def split_serie_when_multiple_windows(test_size, vali_size, ts):
    if vali_size is None:
        train_size = len(ts) - test_size
        return ts[:train_size], ts[train_size:]
    else:
        train_size = len(ts) - vali_size - test_size
        return ts[:train_size], ts[train_size:train_size + vali_size], ts[train_size + vali_size:]


def split_serie(serie, train_perc, vali_perc=0.0):
    train_sample_size = round(len(serie) * train_perc)

    if vali_perc != 0:
        vali_sample_size: int = round(len(serie) * vali_perc)

        train = serie[0:train_sample_size]
        vali = serie[train_sample_size:train_sample_size + vali_sample_size]
        test = serie[(train_sample_size + vali_sample_size):]

        return train, vali, test

    else:
        train = serie[0:train_sample_size]
        test = serie[train_sample_size:]

        return train, test


def stand_interval(series, minimum: float = 0, maximum: float = 1):
    """
        input: serie numpy (n, )
        output:  serie numpy (n, ), scaler (MinMaxScaler object)
    """

    from sklearn.preprocessing import MinMaxScaler

    series = series.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(minimum, maximum)).fit(series)
    series_stand = scaler.transform(series)

    return series_stand, scaler


def transform_data(deployment: str, path_ts: str, ws: int, multiple_windows: bool, train_perc: float,
                   vali_perc: float = 0, *max_ws):
    from pandas import read_csv

    ts = read_csv(path_ts)[deployment].values
    ts_normalized, scaler = stand_interval(ts)
    lags = select_lag_acf(ts_normalized, ws)
    ts_in_windows = create_windows(ts_normalized, ws)

    if multiple_windows:
        vali_size, test_size = get_fixed_sample_sizes(max_ws[0], ts_normalized, train_perc, vali_perc)
        if vali_perc == 0:
            train, test = split_serie_when_multiple_windows(test_size, vali_size, ts_in_windows)

            return train, test, ts, lags, scaler
        else:
            train, vali, test = split_serie_when_multiple_windows(test_size, vali_size, ts_in_windows)

            return train, vali, test, ts, lags, scaler
    else:
        if vali_perc == 0:
            train, test = split_serie(ts_in_windows, train_perc, vali_perc)
            return train, test, ts, lags, scaler
        else:
            train, vali, test = split_serie(ts_in_windows, train_perc, vali_perc)
            return train, vali, test, ts, lags, scaler
