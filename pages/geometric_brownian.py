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
        xaxis_title="Time (Years)",
        yaxis_title="Price",
        template="plotly_white",
        height=500
    )
    
    return fig

def render_page():
    st.title("Geometric Brownian Motion")
    st.markdown("Model stock prices and asset dynamics using stochastic differential equations.")
    
    st.markdown("---")
    
    col_main, col_saved = st.columns([2, 1])
    
    with col_main:
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            S0 = st.number_input("Initial Price (S₀)", value=100.0, min_value=0.01, step=1.0)
            mu = st.number_input("Drift (μ)", value=0.05, step=0.01, format="%.4f", 
                                help="Expected annual return")
            sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01, format="%.4f",
                                   help="Annual volatility")
        
        with col2:
            T = st.number_input("Time Horizon (Years)", value=1.0, min_value=0.1, step=0.1)
            dt = st.number_input("Time Step", value=1/252, min_value=0.001, step=0.001, format="%.6f")
            n_paths = st.number_input("Number of Paths", value=100, min_value=1, max_value=1000, step=10)
        
        col_run, col_save = st.columns([1, 1])
        
        with col_run:
            run_simulation = st.button("Run Simulation", type="primary")
        
        with col_save:
            save_name = st.text_input("Save Name", placeholder="Enter name to save...")
        
        if run_simulation:
            with st.spinner("Running simulation..."):
                t, S = simulate_gbm(S0, mu, sigma, T, dt, int(n_paths))
                
                st.session_state.current_result = {
                    "t": t,
                    "S": S,
                    "params": {
                        "S0": S0,
                        "mu": mu,
                        "sigma": sigma,
                        "T": T,
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
                xaxis_title="Time (Years)",
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
                    st.markdown(f"- T: {params['T']} years")
                    st.markdown(f"- Paths: {params['n_paths']}")
                    
                    st.markdown("**Results:**")
                    stats = item["stats"]
                    st.markdown(f"- Mean: ${stats['mean']:.2f}")
                    st.markdown(f"- Std: ${stats['std']:.2f}")
                    
                    col_load, col_delete = st.columns(2)
                    with col_load:
                        if st.button("Load", key=f"load_{i}"):
                            p = item["params"]
                            t, S = simulate_gbm(p["S0"], p["mu"], p["sigma"], p["T"], p["dt"], p["n_paths"])
                            st.session_state.current_result = {"t": t, "S": S, "params": p}
                            st.rerun()
                    with col_delete:
                        if st.button("Delete", key=f"delete_{i}"):
                            st.session_state.savings["geometric_brownian"].pop(i)
                            st.rerun()
