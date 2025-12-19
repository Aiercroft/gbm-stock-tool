# Finance Recorder

## Overview
An online finance recording tool for running statistical models on financial data. Similar to Excel but optimized for statistical modeling with the ability to save and manage multiple configurations per model.

## Current State
- Geometric Brownian Motion model implemented
- Simulation visualization with Plotly
- Save/load configurations functionality

## Project Structure
- `app.py` - Main application entry point with navigation
- `pages/geometric_brownian.py` - GBM model page with simulation and stats
- `.streamlit/config.toml` - Streamlit server configuration

## Models Available
1. **Geometric Brownian Motion** - Stochastic model for stock price dynamics

## Technical Stack
- Streamlit for UI
- NumPy for numerical computation
- Pandas for data handling
- Plotly for interactive charts
