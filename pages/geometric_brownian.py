import hashlib
import math
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

def simulate_gbm(S0, mu, sigma, T, dt, n_paths):
    """Simulate exact GBM transitions over a horizon measured in years.

    ``dt`` is the target maximum interval in years. The final interval is
    shortened when needed so the simulated path ends exactly at ``T``.
    """
    S0, mu, sigma, T, dt = map(float, (S0, mu, sigma, T, dt))
    n_paths = int(n_paths)

    if not np.isfinite([S0, mu, sigma, T, dt]).all():
        raise ValueError("Simulation inputs must be finite numbers.")
    if S0 <= 0 or sigma < 0 or T < 0 or dt <= 0 or n_paths < 1:
        raise ValueError("S₀ and dt must be positive, σ and T cannot be negative, and at least one path is required.")

    if T == 0:
        return np.array([0.0]), np.full((n_paths, 1), S0, dtype=float)

    step_ratio = T / dt
    nearest_integer = int(round(step_ratio))
    if nearest_integer >= 1 and np.isclose(
        step_ratio, nearest_integer, rtol=1e-12, atol=1e-12
    ):
        n_steps = nearest_integer
    else:
        n_steps = max(1, int(math.ceil(step_ratio)))

    t = np.arange(n_steps + 1, dtype=float) * dt
    t[-1] = T
    step_lengths = np.diff(t)
    dW = np.random.normal(
        0.0,
        np.sqrt(step_lengths)[None, :],
        size=(n_paths, n_steps),
    )
    W = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)],
        axis=1,
    )

    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W

    S = S0 * np.exp(drift + diffusion)
    return t, S

def create_simulation_chart(t, S, title="Simulated price paths"):
    """Create plotly chart for simulation results."""
    fig = go.Figure()
    
    for i in range(min(S.shape[0], 50)):
        fig.add_trace(go.Scatter(
            x=t,
            y=S[i],
            mode='lines',
            opacity=0.22,
            showlegend=False,
            line=dict(color="#287c73", width=1)
        ))
    
    mean_path = np.mean(S, axis=0)
    fig.add_trace(go.Scatter(
        x=t,
        y=mean_path,
        mode='lines',
        name='Mean Path',
        line=dict(color="#bd7557", width=3)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (Price Periods)",
        yaxis_title="Price",
        template="plotly_white",
        height=500,
        paper_bgcolor="#f7f6f0",
        plot_bgcolor="#f7f6f0",
        font=dict(family="DM Sans, sans-serif", color="#24434a"),
        title_font=dict(family="DM Serif Display, serif", size=20, color="#17383e"),
        margin=dict(l=16, r=20, t=56, b=18),
    )
    
    return fig

def calculate_gbm_params(prices, periods_per_year=252):
    """Estimate annual GBM drift and volatility from equally spaced prices."""
    prices = np.asarray(prices, dtype=float)
    periods_per_year = int(periods_per_year)

    if prices.ndim != 1:
        raise ValueError("Prices must be a one-dimensional series.")
    if len(prices) < 3:
        raise ValueError("At least three valid prices are required to estimate volatility.")
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Prices must all be finite and greater than zero.")
    if periods_per_year < 1:
        raise ValueError("Periods per year must be at least 1.")

    log_returns = np.diff(np.log(prices))

    daily_mu = np.mean(log_returns)
    daily_sigma = np.std(log_returns, ddof=1)

    annual_sigma = daily_sigma * np.sqrt(periods_per_year)
    annual_mu = daily_mu * periods_per_year + 0.5 * annual_sigma**2

    return annual_mu, annual_sigma, log_returns


def find_date_column(df):
    """Return the first common date column name, if one exists."""
    return next(
        (
            column
            for column in df.columns
            if str(column).strip().lower() in {"date", "datetime"}
        ),
        None,
    )


def prepare_price_series(df, price_column):
    """Clean selected prices while keeping dates aligned with price values."""
    prices = pd.to_numeric(df[price_column], errors="coerce")
    valid = prices.notna() & np.isfinite(prices) & (prices > 0)
    date_column = find_date_column(df)
    dates = None
    warnings = []

    if date_column is not None:
        parsed_dates = pd.to_datetime(df[date_column], errors="coerce")
        valid &= parsed_dates.notna()
        prepared = pd.DataFrame(
            {
                "Date": parsed_dates.loc[valid],
                "Price": prices.loc[valid].astype(float),
            }
        ).sort_values("Date").reset_index(drop=True)

        if prepared["Date"].duplicated().any():
            raise ValueError("The data has duplicate dates. Keep one price per date.")

        gaps = prepared["Date"].diff().dt.total_seconds().div(86400)
        if (gaps > 3).any():
            warnings.append(
                "Some observations are more than three calendar days apart. "
                "These may be holidays or missing sessions; the estimator treats "
                "each adjacent pair as one price period, so check the data if gaps are unexpected."
            )

        dates = prepared["Date"].to_numpy()
        clean_prices = prepared["Price"].to_numpy(dtype=float)
    else:
        clean_prices = prices.loc[valid].to_numpy(dtype=float)

    dropped_rows = int((~valid).sum())
    if dropped_rows:
        warnings.append(
            f"Ignored {dropped_rows} row(s) with a missing/invalid price"
            + (" or date." if date_column is not None else ".")
        )

    if len(clean_prices) < 3:
        raise ValueError(
            "At least three valid price observations are required to estimate volatility."
        )

    return clean_prices, dates, warnings


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ticker_history(ticker, period):
    """Fetch daily closing prices from Yahoo Finance."""
    history = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=20,
    )

    if history.empty:
        raise ValueError("No price history was returned for this ticker.")

    if isinstance(history.columns, pd.MultiIndex):
        close_level = next(
            (
                level
                for level in range(history.columns.nlevels)
                if "Close" in history.columns.get_level_values(level)
            ),
            None,
        )
        if close_level is None:
            raise ValueError("The provider did not return daily closing prices.")
        history.columns = history.columns.get_level_values(close_level)

    if "Close" not in history.columns:
        raise ValueError("The provider did not return daily closing prices.")

    history.index.name = "Date"
    daily_prices = history[["Close"]].reset_index()
    daily_prices["Date"] = pd.to_datetime(daily_prices["Date"])
    daily_prices = daily_prices.dropna(subset=["Close"])
    daily_prices = daily_prices[daily_prices["Close"] > 0].reset_index(drop=True)

    daily_prices = daily_prices[np.isfinite(daily_prices["Close"])]

    if len(daily_prices) < 3:
        raise ValueError("At least three daily prices are needed for analysis.")

    return daily_prices.reset_index(drop=True)

def render_historical_data_tab():
    """Render the historical data import tab."""
    st.subheader("Historical data")
    st.markdown(
        "Bring in a dated price series, estimate annual GBM parameters, then send them "
        "straight to the simulation."
    )
    data_source = st.radio(
        "Data source",
        ["Ticker lookup", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if data_source == "Ticker lookup":
        with st.form("ticker_history_form"):
            col_ticker, col_period = st.columns([2, 1])
            with col_ticker:
                ticker = st.text_input(
                    "Ticker symbol",
                    placeholder="AAPL",
                    help="Use the symbol used by Yahoo Finance, such as AAPL, MSFT, or BRK-B.",
                ).strip().upper()
            with col_period:
                period = st.selectbox(
                    "History",
                    options=["1y", "2y", "5y", "10y", "max"],
                    index=2,
                    format_func=lambda value: "Maximum available" if value == "max" else value,
                )
            fetch_submitted = st.form_submit_button("Fetch daily prices", type="primary")

        st.caption(
            "Daily Close from Yahoo Finance. Prices are not dividend-adjusted and are not real-time."
        )

        if fetch_submitted:
            if not ticker:
                st.error("Enter a ticker symbol first.")
            else:
                try:
                    with st.spinner(f"Fetching {ticker} price history…"):
                        prices = fetch_ticker_history(ticker, period)
                    st.session_state.historical_df = prices
                    st.session_state.historical_source = f"Yahoo Finance · {ticker}"
                    st.session_state.csv_upload_signature = None
                    st.session_state.pop("calculated_params", None)
                    st.success(f"Loaded {len(prices)} daily Close observations for {ticker}.")
                except Exception:
                    st.error(
                        f"Could not retrieve price history for {ticker}. "
                        "Check the symbol and try again; the data provider may also be temporarily unavailable."
                    )
    else:
        col_upload, col_info = st.columns([2, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                help="CSV should have a column with historical prices",
            )
            if uploaded_file is not None:
                try:
                    file_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
                    if st.session_state.get("csv_upload_signature") != file_signature:
                        df = pd.read_csv(uploaded_file)
                        st.session_state.historical_df = df
                        st.session_state.historical_source = "CSV upload"
                        st.session_state.csv_upload_signature = file_signature
                        st.session_state.pop("calculated_params", None)
                        st.success(f"Loaded {len(df)} rows")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        with col_info:
            st.markdown("**Price series format**")
            st.markdown(
                "A numeric price column, one observation per session. Include a Date column "
                "to sort observations and check for gaps."
            )
    
    if "historical_df" in st.session_state and st.session_state.historical_df is not None:
        df = st.session_state.historical_df.copy()
        date_column = find_date_column(df)
        if date_column is not None:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df = df.sort_values(date_column, na_position="last").reset_index(drop=True)
            st.session_state.historical_df = df
        
        st.markdown("---")
        if st.session_state.get("historical_source"):
            st.caption(f"Loaded from {st.session_state.historical_source}")
        st.markdown(f"**Price series preview** · {len(df)} rows")
        st.dataframe(df, height=300)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            default_idx = 0
            for i, col in enumerate(numeric_cols):
                if col.lower() in ['close', 'adj close', 'adjusted close', 'price']:
                    default_idx = i
                    break
            
            col_select, col_days = st.columns(2)
            
            with col_select:
                price_column = st.selectbox(
                    "Price column",
                    options=numeric_cols,
                    index=default_idx
                )
            
            with col_days:
                periods_per_year = st.number_input(
                    "Annualization basis · periods per year",
                    value=252,
                    min_value=1,
                    max_value=1000,
                    step=1,
                    help="Use 252 for U.S. equities; use 365 for daily crypto data.",
                )
            
            if st.button("Estimate annual parameters", type="primary"):
                try:
                    prices, dates, data_warnings = prepare_price_series(df, price_column)
                    mu, sigma, log_returns = calculate_gbm_params(prices, periods_per_year)
                    st.session_state.calculated_params = {
                        "mu": mu,
                        "sigma": sigma,
                        "S0": prices[-1],
                        "log_returns": log_returns,
                        "prices": prices,
                        "dates": dates,
                        "n_entries": len(prices),
                        "periods_per_year": int(periods_per_year),
                        "price_column": price_column,
                        "data_warnings": data_warnings,
                    }
                    st.success(f"Estimated from {len(prices)} valid price observations.")
                except (TypeError, ValueError) as error:
                    st.session_state.pop("calculated_params", None)
                    st.error(str(error))
            
            previous_calculation = st.session_state.get("calculated_params")
            if previous_calculation and (
                previous_calculation.get("price_column") != price_column
                or previous_calculation.get("periods_per_year")
                != int(periods_per_year)
            ):
                st.session_state.pop("calculated_params", None)
                st.info("The selected price column or annualization basis changed. Recalculate the parameters.")

            if "calculated_params" in st.session_state:
                params = st.session_state.calculated_params
                
                st.markdown("---")
                st.subheader("Estimated parameters")
                st.caption(
                    f"Price column: {params['price_column']} · annualized using "
                    f"{params['periods_per_year']} price periods per year."
                )
                for warning in params.get("data_warnings", []):
                    st.warning(warning)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Annual drift · μ", f"{params['mu']:.4f}")
                with col2:
                    st.metric("Annual volatility · σ", f"{params['sigma']:.4f}")
                with col3:
                    st.metric("Latest Price", f"${params['S0']:.2f}")
                with col4:
                    st.metric("Data Points", f"{params['n_entries']}")
                
                fig_prices = go.Figure()
                x_axis = params['dates'] if params.get('dates') is not None else list(range(len(params['prices'])))
                fig_prices.add_trace(go.Scatter(
                    x=x_axis,
                    y=params['prices'],
                    mode='lines',
                    name='Historical Prices',
                    line=dict(color="#287c73", width=2.5)
                ))
                fig_prices.update_layout(
                    title=f"Historical prices · {params['n_entries']} observations",
                    xaxis_title="Date" if params.get('dates') is not None else "Price Period",
                    yaxis_title="Price",
                    template="plotly_white",
                    height=350,
                    paper_bgcolor="#f7f6f0",
                    plot_bgcolor="#f7f6f0",
                    font=dict(family="DM Sans, sans-serif", color="#24434a"),
                    title_font=dict(family="DM Serif Display, serif", size=20, color="#17383e"),
                    margin=dict(l=16, r=20, t=56, b=18),
                )
                st.plotly_chart(fig_prices)
                
                fig_returns = go.Figure()
                fig_returns.add_trace(go.Histogram(
                    x=params['log_returns'],
                    nbinsx=50,
                    name="Log returns",
                    marker=dict(color="#c47a5d"),
                ))
                fig_returns.update_layout(
                    title="Log-return distribution",
                    xaxis_title="Log Return",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    height=300,
                    paper_bgcolor="#f7f6f0",
                    plot_bgcolor="#f7f6f0",
                    font=dict(family="DM Sans, sans-serif", color="#24434a"),
                    title_font=dict(family="DM Serif Display, serif", size=20, color="#17383e"),
                    margin=dict(l=16, r=20, t=56, b=18),
                )
                st.plotly_chart(fig_returns)
                
                if st.button("Use estimates in simulation", type="primary"):
                    st.session_state.pending_params = {
                        "S0": float(params['S0']),
                        "mu": float(params['mu']),
                        "sigma": float(params['sigma']),
                        "periods_per_year": params["periods_per_year"],
                    }
                    st.session_state.active_tab = "Simulation"
                    st.rerun()
        else:
            st.warning("No numeric columns found in the uploaded file.")

def render_simulation_tab():
    """Render the simulation tab."""
    col_main = st.container()
    col_saved = st.container()

    if "sim_periods_per_year" not in st.session_state:
        st.session_state.sim_periods_per_year = 252
    if "sim_horizon_periods" not in st.session_state:
        st.session_state.sim_horizon_periods = st.session_state.sim_periods_per_year
    if "sim_step_periods" not in st.session_state:
        st.session_state.sim_step_periods = 1.0

    if "pending_params" in st.session_state:
        old_periods_per_year = int(st.session_state.sim_periods_per_year)
        old_horizon_periods = float(st.session_state.sim_horizon_periods)
        horizon_years = old_horizon_periods / old_periods_per_year
        new_periods_per_year = int(
            st.session_state.pending_params.get(
                "periods_per_year", old_periods_per_year
            )
        )
        st.session_state.sim_S0 = st.session_state.pending_params["S0"]
        st.session_state.sim_mu = st.session_state.pending_params["mu"]
        st.session_state.sim_sigma = st.session_state.pending_params["sigma"]
        st.session_state.sim_periods_per_year = new_periods_per_year
        st.session_state.sim_horizon_periods = max(
            1, int(round(horizon_years * new_periods_per_year))
        )
        st.session_state.sim_step_periods = 1.0
        del st.session_state.pending_params

    if "sim_S0" not in st.session_state:
        st.session_state.sim_S0 = 100.0
    if "sim_mu" not in st.session_state:
        st.session_state.sim_mu = 0.05
    if "sim_sigma" not in st.session_state:
        st.session_state.sim_sigma = 0.2

    with col_main:
        st.subheader("Simulation settings")
        st.caption(
            "μ and σ are annualized. Enter the horizon and target step in price periods; "
            "their year equivalents use the periods-per-year basis below."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            S0 = st.number_input("Initial Price (S₀)", min_value=0.01, step=1.0, key="sim_S0")
            mu = st.number_input(
                "Annual drift · μ",
                step=0.01,
                format="%.4f",
                help="The annual drift parameter in dS/S = μdt + σdW.",
                key="sim_mu",
            )
            sigma = st.number_input(
                "Annual volatility · σ",
                min_value=0.0,
                step=0.01,
                format="%.4f",
                help="Annualized standard deviation of returns.",
                key="sim_sigma",
            )
        
        with col2:
            periods_per_year = st.number_input(
                "Annualization basis · periods per year",
                min_value=1,
                max_value=1000,
                step=1,
                key="sim_periods_per_year",
                help="Match this to the basis used to annualize μ and σ (252 for U.S. stocks; often 365 for crypto).",
            )
            horizon_periods = st.number_input(
                "Horizon · price periods",
                min_value=1,
                step=1,
                key="sim_horizon_periods",
            )
            step_periods = st.number_input(
                "Target step · price periods",
                min_value=0.01,
                step=0.25,
                format="%.2f",
                key="sim_step_periods",
                help="Usually 1 for one step per observed price period. The final step is shortened when necessary to end exactly at the horizon.",
            )
            n_paths = st.number_input("Simulated paths", value=20000, min_value=1, step=1000)

        st.caption(
            f"At {int(periods_per_year)} periods/year: horizon = "
            f"{float(horizon_periods) / int(periods_per_year):.3g} years · "
            f"target step = {float(step_periods) / int(periods_per_year):.4g} years. "
            "The final interval shortens when needed to end exactly at the horizon."
        )
        
        col_run, col_save = st.columns([1, 1])
        
        with col_run:
            run_simulation = st.button("Run Simulation", type="primary")
        
        with col_save:
            save_name = st.text_input("Configuration name", placeholder="e.g. Base case · 1y")
        
        if run_simulation:
            with st.spinner("Running simulation..."):
                T = horizon_periods / periods_per_year
                dt = step_periods / periods_per_year
                t_years, S = simulate_gbm(S0, mu, sigma, T, dt, int(n_paths))
                t_periods = t_years * periods_per_year
                
                st.session_state.current_result = {
                    "t": t_periods,
                    "S": S,
                    "params": {
                        "S0": S0,
                        "mu": mu,
                        "sigma": sigma,
                        "horizon_periods": int(horizon_periods),
                        "periods_per_year": int(periods_per_year),
                        "step_periods": float(step_periods),
                        "dt": dt,
                        "n_paths": int(n_paths)
                    }
                }
        
        if "current_result" in st.session_state:
            result = st.session_state.current_result
            result_params = result["params"]
            result_periods_per_year = int(result_params.get("periods_per_year", 252))
            result_horizon_periods = result_params.get(
                "horizon_periods",
                result_params.get("T", 1) * result_periods_per_year,
            )
            result_step_periods = result_params.get(
                "step_periods",
                result_params.get("dt", 1 / result_periods_per_year) * result_periods_per_year,
            )
            st.caption(
                f"Showing {result_params['n_paths']:,} paths · "
                f"{result_horizon_periods:g} price periods "
                f"({result_horizon_periods / result_periods_per_year:.3g} years) · "
                f"target step {result_step_periods:g} price periods · "
                f"annualized on {result_periods_per_year} periods/year."
            )
            fig = create_simulation_chart(result["t"], result["S"])
            st.plotly_chart(fig)
            
            st.subheader("Statistics")
            final_prices = result["S"][:, -1]
            
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
            with stats_col2:
                st.metric("Final-price standard deviation", f"${np.std(final_prices):.2f}")
            with stats_col3:
                st.metric("5th Percentile", f"${np.percentile(final_prices, 5):.2f}")
            with stats_col4:
                st.metric("95th Percentile", f"${np.percentile(final_prices, 95):.2f}")
            
            mean_path = np.mean(result["S"], axis=0)
            fig_mean = go.Figure()
            fig_mean.add_trace(go.Scatter(
                x=result["t"],
                y=mean_path,
                mode='lines',
                name='Average Price',
                line=dict(color="#287c73", width=3)
            ))
            fig_mean.update_layout(
                title="Average Price Over Time",
                xaxis_title="Time (Price Periods)",
                yaxis_title="Price",
                template="plotly_white",
                height=350,
                paper_bgcolor="#f7f6f0",
                plot_bgcolor="#f7f6f0",
                font=dict(family="DM Sans, sans-serif", color="#24434a"),
                title_font=dict(family="DM Serif Display, serif", size=20, color="#17383e"),
                margin=dict(l=16, r=20, t=56, b=18),
            )
            st.plotly_chart(fig_mean)
            
            if save_name and st.button("Save This Configuration"):
                save_entry = {
                    "name": save_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "params": result["params"],
                    "stats": {
                        "mean": float(np.mean(final_prices)),
                        "std": float(np.std(final_prices)),
                        "p5": float(np.percentile(final_prices, 5)),
                        "p95": float(np.percentile(final_prices, 95))
                    }
                }
                st.session_state.savings["geometric_brownian"].append(save_entry)
                st.success(f"Saved '{save_name}'!")
                st.rerun()
    
    with col_saved:
        st.subheader("Saved runs")
        st.caption("Configuration snapshots stay in this browser session.")
        
        saved_items = st.session_state.savings.get("geometric_brownian", [])
        
        if not saved_items:
            st.info("No saved runs yet. Run a simulation and save its settings here.")
        else:
            for i, item in enumerate(saved_items):
                with st.expander(f"{item['name']} ({item['timestamp']})"):
                    st.markdown("**Settings**")
                    params = item["params"]
                    st.markdown(f"- Initial price S₀: {params['S0']}")
                    st.markdown(f"- Annual drift μ: {params['mu']}")
                    st.markdown(f"- Annual volatility σ: {params['sigma']}")
                    config_periods_per_year = int(params.get("periods_per_year", 252))
                    config_horizon_periods = params.get(
                        "horizon_periods",
                        params.get(
                            "T_days",
                            params.get("T", 0) * config_periods_per_year,
                        ),
                    )
                    st.markdown(
                        f"- Horizon: {config_horizon_periods} price periods "
                        f"({config_horizon_periods / config_periods_per_year:.2f} years)"
                    )
                    st.markdown(f"- Target step: {params.get('step_periods', 1)} price periods")
                    st.markdown(f"- Annualization basis: {config_periods_per_year} periods/year")
                    st.markdown(f"- Simulated paths: {params['n_paths']}")
                    
                    st.markdown("**Results:**")
                    stats = item["stats"]
                    st.markdown(f"- Mean: ${stats['mean']:.2f}")
                    st.markdown(f"- Std: ${stats['std']:.2f}")
                    
                    col_load, col_delete = st.columns(2)
                    with col_load:
                        if st.button("Load", key=f"load_{i}"):
                            p = item["params"]
                            loaded_periods_per_year = int(p.get("periods_per_year", 252))
                            loaded_horizon_periods = p.get(
                                "horizon_periods",
                                p.get("T_days", p.get("T", 1) * loaded_periods_per_year),
                            )
                            loaded_T = loaded_horizon_periods / loaded_periods_per_year
                            loaded_dt = p.get(
                                "dt",
                                p.get("step_periods", 1) / loaded_periods_per_year,
                            )
                            t_years, S = simulate_gbm(
                                p["S0"],
                                p["mu"],
                                p["sigma"],
                                loaded_T,
                                loaded_dt,
                                p["n_paths"],
                            )
                            st.session_state.current_result = {
                                "t": t_years * loaded_periods_per_year,
                                "S": S,
                                "params": p,
                            }
                            st.rerun()
                    with col_delete:
                        if st.button("Delete", key=f"delete_{i}"):
                            st.session_state.savings["geometric_brownian"].pop(i)
                            st.rerun()

def render_page():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');
        :root {
            --paper: #f3f1e9;
            --surface: #f9f8f3;
            --ink: #18383e;
            --muted: #617578;
            --line: #d7ddd5;
            --teal: #287c73;
            --teal-deep: #1b5c56;
            --clay: #bd7557;
        }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(243, 241, 233, .88); }
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        button[aria-label="Open sidebar"] { display: none !important; }
        .block-container {
            max-width: 1080px !important;
            margin-left: auto;
            margin-right: auto;
            padding-top: 2.4rem;
            padding-bottom: 4rem;
            padding-left: clamp(1rem, 4vw, 4.4rem);
            padding-right: clamp(1rem, 4vw, 4.4rem);
        }
        [data-testid="stSidebar"] { display: none; }
        .gbm-masthead {
            position: relative;
            padding: .45rem 0 1.35rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--line);
            text-align: center;
        }
        .gbm-heading {
            margin: .2rem 0 .35rem; color: var(--ink);
            font: 400 clamp(2.3rem, 5vw, 4rem)/1.05 'DM Serif Display', serif;
            letter-spacing: -.035em;
        }
        .gbm-intro {
            max-width: 42rem; margin: 0 auto; color: var(--muted);
            font-size: 1rem; line-height: 1.65;
        }
        h2, h3 { color: var(--ink) !important; }
        h2 { font: 400 1.8rem 'DM Serif Display', serif !important; }
        h3 { font: 400 1.45rem 'DM Serif Display', serif !important; }
        [data-testid="stCaptionContainer"] { color: var(--muted); }
        [data-testid="stMetric"] {
            padding: 1rem 1.1rem; border: 1px solid var(--line);
            border-radius: 10px; background: rgba(249, 248, 243, .82);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); font-size: .78rem; }
        [data-testid="stMetricValue"] {
            color: var(--ink); font: 500 1.55rem 'DM Mono', monospace;
        }
        [data-testid="stPlotlyChart"] {
            padding: .45rem; border: 1px solid var(--line);
            border-radius: 12px; background: var(--surface);
        }
        .stRadio [role="radiogroup"] {
            display: inline-flex; gap: .35rem; padding: .32rem;
            border: 1px solid var(--line); border-radius: 999px;
            background: #e9eae2;
            margin: 0 auto;
        }
        [data-testid="stRadio"] { display: flex; justify-content: center; }
        .stRadio [role="radiogroup"] label {
            padding: .52rem 1rem; border-radius: 999px; color: var(--muted);
            transition: background .18s ease, color .18s ease;
        }
        .stRadio [role="radiogroup"] label:has(input:checked) {
            color: #f7f6f0; background: var(--teal-deep);
        }
        .stRadio [role="radiogroup"] label p {
            font-size: .9rem;
            color: inherit;
        }
        .stRadio [role="radiogroup"] label:has(input:checked) p {
            color: #f7f6f0 !important;
        }
        .stButton > button, [data-testid="stFormSubmitButton"] button {
            min-height: 2.75rem; padding: .55rem 1.05rem;
            border: 1px solid var(--teal-deep); border-radius: 7px;
            background: var(--teal-deep); color: #f7f6f0;
            font-weight: 600; transition: transform .16s ease, background .16s ease;
        }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
            border-color: var(--teal); background: var(--teal); color: #fff;
            transform: translateY(-1px);
        }
        input, textarea, [data-baseweb="select"] > div {
            border-radius: 7px !important;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line); border-radius: 9px; overflow: hidden;
        }
        [data-testid="stExpander"] {
            border-color: var(--line); border-radius: 9px;
            background: rgba(249, 248, 243, .78);
        }
        [data-testid="stAlert"] { border-radius: 9px; }
        hr { border-color: var(--line); }
        @media (max-width: 767px) {
            .block-container { padding-top: 1.3rem; }
            .gbm-masthead { padding-bottom: 1.25rem; }
            .gbm-heading { font-size: clamp(2.25rem, 12vw, 3.15rem); }
            .stRadio [role="radiogroup"] { width: 100%; }
            .stRadio [role="radiogroup"] label {
                justify-content: center;
                flex: 1;
                padding-left: .65rem;
                padding-right: .65rem;
                white-space: nowrap;
            }
            [data-testid="stMetric"] { padding: .8rem; }
            [data-testid="stMetricValue"] { font-size: 1.2rem; }
        }
        </style>
        <div class="gbm-masthead">
            <h1 class="gbm-heading">Geometric Brownian Motion</h1>
            <p class="gbm-intro">Turn a price series into annual drift and volatility, then explore the distribution of possible paths.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Simulation"
    
    tab_options = ["Simulation", "Historical Data"]
    selected_tab = st.radio(
        "Workspace view",
        tab_options,
        index=tab_options.index(st.session_state.active_tab),
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if selected_tab != st.session_state.active_tab:
        st.session_state.active_tab = selected_tab
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.active_tab == "Simulation":
        render_simulation_tab()
    else:
        render_historical_data_tab()
