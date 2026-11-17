from pathlib import Path
import io
import warnings
import zipfile

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller, kpss


def load_online_retail(local_path="Online Retail.xlsx"):
    local_path = Path(local_path)
    if local_path.exists():
        return pd.read_excel(local_path)

    url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
    try:
        import requests

        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            xlsx_name = next(
                name for name in archive.namelist()
                if name.lower().endswith(".xlsx")
            )
            with archive.open(xlsx_name) as file:
                return pd.read_excel(file)
    except Exception as exc:
        raise RuntimeError(
            "Could not load Online Retail automatically. "
            "Place Online Retail.xlsx next to the notebook."
        ) from exc


def make_daily_sales(
    df,
    stock_code,
    date_col="InvoiceDate",
    quantity_col="Quantity",
    stock_col="StockCode",
    positive_only=True,
):
    data = df[df[stock_col].astype(str).eq(str(stock_code))].copy()
    if positive_only:
        data = data[data[quantity_col] > 0]

    sales = (
        data
        .set_index(date_col)[[quantity_col]]
        .sort_index()
        .resample("D")
        .sum()
        .rename(columns={quantity_col: "Quantity"})
    )
    sales.index.name = date_col
    return sales


def aggregate_sales(sales):
    return {
        "Daily": sales.resample("D").sum(),
        "Weekly": sales.resample("W").sum(),
        "Monthly": sales.resample("MS").sum(),
    }


def zero_period_summary(aggregated_sales):
    missing_sales = {
        name: series[series["Quantity"].eq(0)].index
        for name, series in aggregated_sales.items()
    }
    summary = pd.DataFrame({
        "Aggregation": missing_sales.keys(),
        "Missing periods count": [len(dates) for dates in missing_sales.values()],
        "Missing dates": [
            ", ".join(dates.strftime("%Y-%m-%d"))
            for dates in missing_sales.values()
        ],
    })
    return missing_sales, summary


def weekday_zero_stats(missing_daily_dates):
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    missing_daily_dates = pd.Series(missing_daily_dates, name="InvoiceDate")
    return (
        missing_daily_dates
        .dt.day_name()
        .value_counts()
        .reindex(weekday_order, fill_value=0)
        .rename_axis("Weekday")
        .reset_index(name="Missing days count")
    )


def detect_iqr_anomalies(
    sales_daily,
    window=14,
    iqr_multiplier=3,
    holiday_start="2010-12-24",
    holiday_end="2011-01-03",
):
    daily_anomalies = sales_daily.copy()
    holiday_start = pd.Timestamp(holiday_start)
    holiday_end = pd.Timestamp(holiday_end)

    is_zero = daily_anomalies["Quantity"].eq(0)
    is_saturday = daily_anomalies.index.dayofweek == 5
    is_holiday_period = (
        (daily_anomalies.index >= holiday_start)
        & (daily_anomalies.index <= holiday_end)
    )
    ignored_zero_mask = is_zero & (is_saturday | is_holiday_period)

    daily_anomalies["ignored_zero"] = ignored_zero_mask
    daily_anomalies["Quantity_for_iqr"] = daily_anomalies["Quantity"].mask(
        ignored_zero_mask
    )
    has_full_past_window = (
        daily_anomalies["Quantity"]
        .shift(1)
        .rolling(window=window, min_periods=window)
        .count()
        .eq(window)
    )
    past_window = (
        daily_anomalies["Quantity_for_iqr"]
        .shift(1)
        .rolling(window=window, min_periods=1)
    )

    daily_anomalies["q25"] = past_window.quantile(0.25)
    daily_anomalies["q75"] = past_window.quantile(0.75)
    daily_anomalies["iqr"] = daily_anomalies["q75"] - daily_anomalies["q25"]
    daily_anomalies["lower_bound"] = (
        daily_anomalies["q25"] - iqr_multiplier * daily_anomalies["iqr"]
    )
    daily_anomalies["upper_bound"] = (
        daily_anomalies["q75"] + iqr_multiplier * daily_anomalies["iqr"]
    )
    daily_anomalies["is_anomaly"] = (
        has_full_past_window
        & ~ignored_zero_mask
        & (
            daily_anomalies["Quantity"].lt(daily_anomalies["lower_bound"])
            | daily_anomalies["Quantity"].gt(daily_anomalies["upper_bound"])
        )
    )

    ignored_zero_points = daily_anomalies[ignored_zero_mask].reset_index()[
        ["InvoiceDate", "Quantity"]
    ]
    anomaly_points = daily_anomalies[daily_anomalies["is_anomaly"]].reset_index()[
        ["InvoiceDate", "Quantity", "q25", "q75", "lower_bound", "upper_bound"]
    ]
    return daily_anomalies, anomaly_points, ignored_zero_points


def make_clean_sales(daily_anomalies):
    keep_zero_mask = (
        daily_anomalies["Quantity"].ne(0)
        | daily_anomalies["ignored_zero"]
    )
    dropped_zero_points = daily_anomalies[
        daily_anomalies["Quantity"].eq(0)
        & ~daily_anomalies["ignored_zero"]
    ].reset_index()[["InvoiceDate", "Quantity"]]

    sales_clean = daily_anomalies.loc[
        ~daily_anomalies["is_anomaly"] & keep_zero_mask,
        ["Quantity"],
    ].copy()
    sales_clean.index.name = "InvoiceDate"
    return sales_clean, dropped_zero_points


def sales_statistics(sales_clean):
    return pd.DataFrame({
        "Metric": ["Mean", "Variance", "Standard deviation"],
        "Value": [
            sales_clean["Quantity"].mean(),
            sales_clean["Quantity"].var(),
            sales_clean["Quantity"].std(),
        ],
    })


def average_sales_by_calendar(sales_clean):
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }
    weekday_names = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday",
    }

    monthly = (
        sales_clean
        .assign(Month=sales_clean.index.month.map(month_names))
        .groupby("Month", sort=False)["Quantity"]
        .mean()
        .reindex(month_names.values())
        .dropna()
        .reset_index(name="Average sales")
    )
    weekdays = (
        sales_clean
        .assign(Weekday=sales_clean.index.dayofweek.map(weekday_names))
        .groupby("Weekday", sort=False)["Quantity"]
        .mean()
        .reindex(weekday_names.values())
        .dropna()
        .reset_index(name="Average sales")
    )
    return monthly, weekdays


def make_regular_daily_series(sales_clean, fill_method="interpolate"):
    series = sales_clean.asfreq("D")["Quantity"]
    if fill_method == "zero":
        return series.fillna(0.0)
    if fill_method == "interpolate":
        return series.interpolate(method="time").bfill().ffill()
    raise ValueError("fill_method must be 'zero' or 'interpolate'.")


def fft_spectrum(sales_clean):
    series = make_regular_daily_series(sales_clean, fill_method="zero")
    n = len(series)
    y = series.to_numpy(dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    y = y - y.mean()

    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1)
    amplitude = np.abs(spectrum) / n
    if n % 2 == 0:
        amplitude[1:-1] *= 2
    else:
        amplitude[1:] *= 2

    dominant_idx = np.argmax(amplitude[1:]) + 1
    weekly_idx = np.argmin(np.abs(freqs - 1 / 7))
    return {
        "series": series,
        "freqs": freqs,
        "amplitude": amplitude,
        "dominant_frequency": freqs[dominant_idx],
        "dominant_period": 1 / freqs[dominant_idx],
        "dominant_amplitude": amplitude[dominant_idx],
        "weekly_amplitude": amplitude[weekly_idx],
        "weekly_period": 1 / freqs[weekly_idx],
    }


def cwt_scalogram(sales_clean, wavelet="cmor1.5-1.0", min_period=1, max_period=20):
    series = make_regular_daily_series(sales_clean, fill_method="zero")
    y = series.to_numpy(dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    y = y - y.mean()

    target_freqs = np.linspace(1 / max_period, 1 / min_period, 100)
    scales = pywt.frequency2scale(wavelet, target_freqs)
    coefficients, frequencies = pywt.cwt(
        y,
        scales,
        wavelet,
        sampling_period=1,
    )
    periods = 1 / frequencies
    amplitude = np.abs(coefficients)
    power = amplitude ** 2
    max_power_idx = np.unravel_index(np.argmax(power), power.shape)

    return {
        "series": series,
        "periods": periods,
        "power": power,
        "amplitude": amplitude,
        "max_amplitude": amplitude[max_power_idx],
        "max_power": power[max_power_idx],
        "max_period": periods[max_power_idx[0]],
        "max_date": series.index[max_power_idx[1]],
    }


def mstl_decomposition(sales_clean, periods=(7, 30)):
    series = make_regular_daily_series(sales_clean, fill_method="interpolate")
    result = MSTL(series, periods=list(periods)).fit()

    seasonal = result.seasonal
    if isinstance(seasonal, pd.Series):
        seasonal = seasonal.to_frame(name=f"seasonal_{periods[0]}")

    seasonal_strengths = {}
    for column in seasonal.columns:
        strength = max(
            0,
            1 - np.var(result.resid) / np.var(seasonal[column] + result.resid),
        )
        seasonal_strengths[column] = strength

    trend_strength = max(
        0,
        1 - np.var(result.resid) / np.var(result.trend + result.resid),
    )

    components = pd.concat(
        [
            series.rename("observed"),
            result.trend.rename("trend"),
            seasonal,
            result.resid.rename("resid"),
        ],
        axis=1,
    )
    return result, components, seasonal_strengths, trend_strength


def stationarity_tests(sales_clean):
    series = make_regular_daily_series(sales_clean, fill_method="interpolate")
    adf_result = adfuller(series, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_result = kpss(series, regression="c", nlags="auto")

    return pd.DataFrame([
        {
            "Series": "Original",
            "Test": "ADF",
            "H0": "unit root, non-stationary",
            "Statistic": adf_result[0],
            "p-value": adf_result[1],
            "Conclusion": "stationary" if adf_result[1] < 0.05 else "non-stationary",
        },
        {
            "Series": "Original",
            "Test": "KPSS",
            "H0": "stationary around constant level",
            "Statistic": kpss_result[0],
            "p-value": kpss_result[1],
            "Conclusion": "stationary" if kpss_result[1] >= 0.05 else "non-stationary",
        },
    ])


def plot_clean_sales(sales_clean):
    zero_points = sales_clean[sales_clean["Quantity"].eq(0)]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sales_clean.index, sales_clean["Quantity"], label="Итоговый ряд")
    if not zero_points.empty:
        ax.scatter(
            zero_points.index,
            zero_points["Quantity"],
            color="red",
            s=35,
            label="Нулевые значения",
        )
    ax.set_title("Итоговый временной ряд после предобработки")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Продажи")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_calendar_averages(monthly_average, weekday_average):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    axes[0].bar(
        monthly_average["Month"],
        monthly_average["Average sales"],
        color="steelblue",
    )
    axes[0].set_title("Средние продажи по месяцам")
    axes[0].set_xlabel("Месяц")
    axes[0].set_ylabel("Средние продажи")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        weekday_average["Weekday"],
        weekday_average["Average sales"],
        color="seagreen",
    )
    axes[1].set_title("Средние продажи по дням недели")
    axes[1].set_xlabel("День недели")
    axes[1].set_ylabel("Средние продажи")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_fft_spectrum(fft_result):
    freqs = fft_result["freqs"]
    amplitude = fft_result["amplitude"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(freqs[1:], amplitude[1:])
    ax.axvline(
        1 / 7,
        ls="--",
        c="g",
        alpha=0.5,
        label="Недельная частота",
    )
    ax.set_title("Амплитудный спектр продаж")
    ax.set_xlabel("Частота, циклов в день")
    ax.set_ylabel("Амплитуда")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_cwt_scalogram(cwt_result):
    series = cwt_result["series"]
    periods = cwt_result["periods"]
    power = cwt_result["power"]
    y = series.to_numpy(dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    y = y - y.mean()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True,
        layout="constrained",
    )
    axes[0].plot(series.index, y)
    axes[0].set_title("Ежедневные продажи: колебания относительно среднего")
    axes[0].set_ylabel("Отклонение продаж")
    axes[0].grid(alpha=0.3)

    mesh = axes[1].pcolormesh(
        series.index,
        periods,
        power,
        shading="auto",
        cmap="viridis",
    )
    axes[1].set_title("Скалограмма CWT")
    axes[1].set_ylabel("Период колебаний, дни")
    axes[1].set_xlabel("Дата")
    axes[1].axhline(
        7,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Неделя - 7 дней",
    )
    axes[1].legend()
    fig.colorbar(mesh, ax=axes[1], label="Мощность |CWT|^2")
    return fig, axes


def plot_mstl_components(mstl_components):
    seasonal_columns = [
        column for column in mstl_components.columns
        if column.startswith("seasonal_")
    ]

    n_rows = 3 + len(seasonal_columns)
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 3 * n_rows), sharex=True)

    axes[0].plot(mstl_components.index, mstl_components["observed"])
    axes[0].set_title("MSTL: наблюдаемый ряд")
    axes[0].set_ylabel("Продажи")

    axes[1].plot(
        mstl_components.index,
        mstl_components["trend"],
        color="tab:orange",
    )
    axes[1].set_title("MSTL: тренд")
    axes[1].set_ylabel("Продажи")

    for idx, column in enumerate(seasonal_columns, start=2):
        period = column.replace("seasonal_", "")
        axes[idx].plot(mstl_components.index, mstl_components[column])
        axes[idx].set_title(f"MSTL: сезонность {period} дней")
        axes[idx].set_ylabel("Отклонение")

    resid_ax = axes[-1]
    resid_ax.plot(mstl_components.index, mstl_components["resid"], color="tab:red")
    resid_ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    resid_ax.set_title("MSTL: остатки")
    resid_ax.set_xlabel("Дата")
    resid_ax.set_ylabel("Остаток")

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_stationarity_acf(sales_clean, lags=60):
    series = make_regular_daily_series(sales_clean, fill_method="interpolate")
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_acf(series, lags=lags, ax=ax)
    ax.set_title("ACF исходного ряда")
    ax.set_xlabel("Лаг, дни")
    ax.set_ylabel("Автокорреляция")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_pipeline_results(results):
    figures = {}
    figures["clean_sales"] = plot_clean_sales(results["sales_clean"])
    figures["calendar_averages"] = plot_calendar_averages(
        results["monthly_average"],
        results["weekday_average"],
    )
    figures["fft"] = plot_fft_spectrum(results["fft"])
    figures["cwt"] = plot_cwt_scalogram(results["cwt"])
    figures["mstl"] = plot_mstl_components(results["mstl_components"])
    figures["acf"] = plot_stationarity_acf(results["sales_clean"])
    return figures


def run_analysis_pipeline(
    df,
    stock_code="85123A",
    window=14,
    iqr_multiplier=3,
    holiday_start="2010-12-24",
    holiday_end="2011-01-03",
):
    sales = make_daily_sales(df, stock_code=stock_code, positive_only=True)
    aggregated = aggregate_sales(sales)
    missing_sales, missing_summary = zero_period_summary(aggregated)
    weekday_missing = weekday_zero_stats(missing_sales["Daily"])
    daily_anomalies, anomaly_points, ignored_zero_points = detect_iqr_anomalies(
        aggregated["Daily"],
        window=window,
        iqr_multiplier=iqr_multiplier,
        holiday_start=holiday_start,
        holiday_end=holiday_end,
    )
    sales_clean, dropped_zero_points = make_clean_sales(daily_anomalies)
    monthly_average, weekday_average = average_sales_by_calendar(sales_clean)
    fft = fft_spectrum(sales_clean)
    cwt = cwt_scalogram(sales_clean)
    mstl_result, mstl_components, seasonal_strengths, trend_strength = (
        mstl_decomposition(sales_clean)
    )
    stationarity = stationarity_tests(sales_clean)

    return {
        "sales": sales,
        "aggregated": aggregated,
        "missing_summary": missing_summary,
        "weekday_missing": weekday_missing,
        "daily_anomalies": daily_anomalies,
        "anomaly_points": anomaly_points,
        "ignored_zero_points": ignored_zero_points,
        "sales_clean": sales_clean,
        "dropped_zero_points": dropped_zero_points,
        "statistics": sales_statistics(sales_clean),
        "monthly_average": monthly_average,
        "weekday_average": weekday_average,
        "fft": fft,
        "cwt": cwt,
        "mstl_result": mstl_result,
        "mstl_components": mstl_components,
        "seasonal_strengths": seasonal_strengths,
        "trend_strength": trend_strength,
        "stationarity": stationarity,
    }

