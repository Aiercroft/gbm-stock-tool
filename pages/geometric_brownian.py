import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def simulate_gbm(S0, mu, sigma, T, dt, n_paths):
    """Simulate Geometric Brownian Motion paths."""
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)
    
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    W = np.cumsum(dW, axis=1)
    W = np.hstack([np.zeros((n_paths, 1)), W])
    
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W
    
    S = S0 * np.exp(drift + diffusion)
    return t, S

def create_simulation_chart(t, S, title="GBM Simulation"):
    """Create plotly chart for simulation results."""
    fig = go.Figure()
    
    for i in range(min(S.shape[0], 50)):
        fig.add_trace(go.Scatter(
            x=t,
            y=S[i],
            mode='lines',
            opacity=0.6,
            showlegend=False,
            line=dict(width=1)
        ))
    
    mean_path = np.mean(S, axis=0)
    fig.add_trace(go.Scatter(
        x=t,
        y=mean_path,
        mode='lines',
        name='Mean Path',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (Days)",
        yaxis_title="Price",
        template="plotly_white",
        height=500
    )
    
    return fig

def calculate_gbm_params(prices, trading_days_per_year=252):
    """Calculate drift and volatility from historical prices."""
    prices = np.array(prices)
    log_returns = np.diff(np.log(prices))
    
    daily_mu = np.mean(log_returns)
    daily_sigma = np.std(log_returns, ddof=1)
    
    annual_sigma = daily_sigma * np.sqrt(trading_days_per_year)
    annual_mu = daily_mu * trading_days_per_year + 0.5 * annual_sigma**2
    
    return annual_mu, annual_sigma, log_returns

def render_historical_data_tab():
    """Render the historical data import tab."""
    st.subheader("Import Historical Data")
    st.markdown("Upload price data to calculate drift (μ) and volatility (σ) parameters.")
    st.markdown("Get free historical data from [Stooq](https://stooq.com/)")
    
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="CSV should have a column with historical prices"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.historical_df = df
                st.success(f"Loaded {len(df)} rows")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col_info:
        st.markdown("**Expected format:**")
        st.markdown("- CSV with price column")
        st.markdown("- Prices in chronological order")
        st.markdown("- One price per trading day")
    
    if "historical_df" in st.session_state and st.session_state.historical_df is not None:
        df = st.session_state.historical_df
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            st.session_state.historical_df = df
        
        st.markdown("---")
        st.markdown(f"**Data Preview:** ({len(df)} total entries)")
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
                    "Select price column",
                    options=numeric_cols,
                    index=default_idx
                )
            
            with col_days:
                trading_days = st.number_input(
                    "Trading days per year",
                    value=252,
                    min_value=1,
                    step=1
                )
            
            if st.button("Calculate Parameters", type="primary"):
                prices = df[price_column].dropna().values
                dates = df['Date'].values if 'Date' in df.columns else None
                
                if len(prices) < 2:
                    st.error("Need at least 2 price points")
                else:
                    mu, sigma, log_returns = calculate_gbm_params(prices, trading_days)
                    
                    st.session_state.calculated_params = {
                        "mu": mu,
                        "sigma": sigma,
                        "S0": prices[-1],
                        "log_returns": log_returns,
                        "prices": prices,
                        "dates": dates,
                        "n_entries": len(prices)
                    }
                    st.success(f"Calculated from {len(prices)} price entries")
            
            if "calculated_params" in st.session_state:
                params = st.session_state.calculated_params
                
                st.markdown("---")
                st.subheader("Calculated Parameters")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Drift (μ)", f"{params['mu']:.4f}")
                with col2:
                    st.metric("Volatility (σ)", f"{params['sigma']:.4f}")
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
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_prices.update_layout(
                    title=f"Historical Price Data ({params['n_entries']} entries)",
                    xaxis_title="Date" if params.get('dates') is not None else "Trading Day",
                    yaxis_title="Price",
                    template="plotly_white",
                    height=350
                )
                st.plotly_chart(fig_prices)
                
                fig_returns = go.Figure()
                fig_returns.add_trace(go.Histogram(
                    x=params['log_returns'],
                    nbinsx=50,
                    name='Log Returns'
                ))
                fig_returns.update_layout(
                    title="Distribution of Log Returns",
                    xaxis_title="Log Return",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig_returns)
                
                if st.button("Use These Parameters in Simulation"):
                    st.session_state.use_calculated_params = True
                    st.success("Parameters saved! Switch to the Simulation tab to use them.")
        else:
            st.warning("No numeric columns found in the uploaded file.")

def render_simulation_tab():
    """Render the simulation tab."""
    col_main, col_saved = st.columns([2, 1])
    
    default_mu = 0.05
    default_sigma = 0.2
    default_S0 = 100.0
    
    if st.session_state.get("use_calculated_params") and "calculated_params" in st.session_state:
        params = st.session_state.calculated_params
        default_mu = params["mu"]
        default_sigma = params["sigma"]
        default_S0 = params["S0"]
        st.session_state.use_calculated_params = False
    
    with col_main:
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            S0 = st.number_input("Initial Price (S₀)", value=default_S0, min_value=0.01, step=1.0)
            mu = st.number_input("Drift (μ)", value=default_mu, step=0.01, format="%.4f", 
                                help="Expected annual return")
            sigma = st.number_input("Volatility (σ)", value=default_sigma, min_value=0.01, step=0.01, format="%.4f",
                                   help="Annual volatility")
        
        with col2:
            T_days = st.number_input("Time Horizon (Days)", value=252, min_value=1, step=1)
            dt = st.number_input("Time Step", value=1/252, min_value=0.001, step=0.001, format="%.6f")
            n_paths = st.number_input("Number of Paths", value=20000, min_value=1, step=1000)
        
        col_run, col_save = st.columns([1, 1])
        
        with col_run:
            run_simulation = st.button("Run Simulation", type="primary")
        
        with col_save:
            save_name = st.text_input("Save Name", placeholder="Enter name to save...")
        
        if run_simulation:
            with st.spinner("Running simulation..."):
                T = T_days / 252
                t, S = simulate_gbm(S0, mu, sigma, T, dt, int(n_paths))
                t_days = t * 252
                
                st.session_state.current_result = {
                    "t": t_days,
                    "S": S,
                    "params": {
                        "S0": S0,
                        "mu": mu,
                        "sigma": sigma,
                        "T_days": T_days,
                        "dt": dt,
                        "n_paths": int(n_paths)
                    }
                }
        
        if "current_result" in st.session_state:
            result = st.session_state.current_result
            
            fig = create_simulation_chart(result["t"], result["S"])
            st.plotly_chart(fig)
            
            st.subheader("Statistics")
            final_prices = result["S"][:, -1]
            
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
            with stats_col2:
                st.metric("Std Dev", f"${np.std(final_prices):.2f}")
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
                line=dict(color='#1f77b4', width=3)
            ))
            fig_mean.update_layout(
                title="Average Price Over Time",
                xaxis_title="Time (Days)",
                yaxis_title="Price",
                template="plotly_white",
                height=350
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
        st.subheader("Saved Configurations")
        
        saved_items = st.session_state.savings.get("geometric_brownian", [])
        
        if not saved_items:
            st.info("No saved configurations yet. Run a simulation and save it.")
        else:
            for i, item in enumerate(saved_items):
                with st.expander(f"{item['name']} ({item['timestamp']})"):
                    st.markdown("**Parameters:**")
                    params = item["params"]
                    st.markdown(f"- S₀: {params['S0']}")
                    st.markdown(f"- μ: {params['mu']}")
                    st.markdown(f"- σ: {params['sigma']}")
                    st.markdown(f"- T: {params.get('T_days', params.get('T', 0) * 252)} days")
                    st.markdown(f"- Paths: {params['n_paths']}")
                    
                    st.markdown("**Results:**")
                    stats = item["stats"]
                    st.markdown(f"- Mean: ${stats['mean']:.2f}")
                    st.markdown(f"- Std: ${stats['std']:.2f}")
                    
                    col_load, col_delete = st.columns(2)
                    with col_load:
                        if st.button("Load", key=f"load_{i}"):
                            p = item["params"]
                            T_days = p.get('T_days', p.get('T', 1) * 252)
                            T = T_days / 252
                            t, S = simulate_gbm(p["S0"], p["mu"], p["sigma"], T, p["dt"], p["n_paths"])
                            t_days = t * 252
                            st.session_state.current_result = {"t": t_days, "S": S, "params": p}
                            st.rerun()
                    with col_delete:
                        if st.button("Delete", key=f"delete_{i}"):
                            st.session_state.savings["geometric_brownian"].pop(i)
                            st.rerun()

def render_page():
    st.title("Geometric Brownian Motion")
    st.markdown("Model stock prices and asset dynamics using stochastic differential equations.")
    
    st.markdown("---")
    
    tab_sim, tab_data = st.tabs(["Simulation", "Historical Data"])
    
    with tab_sim:
        render_simulation_tab()
    
    with tab_data:
        render_historical_data_tab()
