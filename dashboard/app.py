import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import DataPreprocessor
from src.eda import ExploratoryAnalysis
from src.models.lstm_model import LSTMForecaster
from src.models.arima_xgboost import HybridArimaXGBoost
from src.models.model_comparison import ModelComparison
from src.inventory.optimization import InventoryOptimization

# Configure page
st.set_page_config(
    page_title="AI Demand Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-title { font-size: 40px; color: #1f77b4; text-align: center; margin-bottom: 10px; }
    .section-title { font-size: 24px; color: #ff7f0e; margin-top: 20px; border-bottom: 2px solid #ff7f0e; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = DataPreprocessor()
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# Main title
st.markdown("<div class='main-title'>🤖 AI-Based Demand Forecasting System</div>", unsafe_allow_html=True)
st.markdown("**Intelligent demand forecasting and inventory optimization powered by LSTM and Hybrid ARIMA+XGBoost**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    forecast_steps = st.slider("Forecast Period (days)", 1, 90, 30)
    lead_time = st.slider("Lead Time (days)", 1, 30, 7)
    service_level = st.slider("Service Level (%)", 80, 99, 95) / 100
    
    st.divider()
    st.header("Model Parameters")
    
    lstm_epochs = st.slider("LSTM Epochs", 10, 100, 50)
    lstm_batch = st.slider("LSTM Batch Size", 8, 64, 32)
    arima_order = st.text_input("ARIMA Order (p,d,q)", value="1,1,1")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📤 Data Upload", "📊 EDA Analysis", "🤖 Model Training", "📈 Forecasts", "📦 Inventory"]
)

# TAB 1: DATA UPLOAD
with tab1:
    st.markdown("<div class='section-title'>Data Upload & Preprocessing</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = st.session_state.preprocessor.load_data(uploaded_file)
            st.session_state.data = df
            
            st.success("✅ File uploaded successfully!")
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Date Range", f"{df.iloc[0, 0]} to {df.iloc[-1, 0]}")
            
            # Show raw data
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Configure preprocessing
            st.subheader("Configure Columns")
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Select Date Column", df.columns, key="date_col")
            with col2:
                sales_col = st.selectbox("Select Sales Column", df.columns, key="sales_col")
            
            # Preprocess button
            if st.button("🔄 Preprocess Data", key="preprocess_btn"):
                with st.spinner("Preprocessing data..."):
                    df_clean = st.session_state.preprocessor.clean_data(df, date_col, sales_col)
                    df_clean = st.session_state.preprocessor.create_time_features(df_clean, date_col)
                    df_clean = st.session_state.preprocessor.create_lag_features(df_clean, sales_col)
                    
                    st.session_state.data = df_clean
                    
                    st.success("✅ Data preprocessed successfully!")
                    st.subheader("Processed Data Preview")
                    st.dataframe(df_clean.head(10), use_container_width=True)
                    
                    st.info(f"📌 Data shape after preprocessing: {df_clean.shape}")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# TAB 2: EDA ANALYSIS
with tab2:
    st.markdown("<div class='section-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        df = st.session_state.data
        date_col = st.selectbox("Select Date Column", df.columns, key="eda_date")
        sales_col = st.selectbox("Select Sales Column", df.columns, key="eda_sales")
        
        eda = ExploratoryAnalysis()
        
        # Summary Statistics
        st.subheader("📊 Summary Statistics")
        stats = eda.generate_summary_statistics(df, sales_col)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{stats['Mean']:.2f}")
        with col2:
            st.metric("Std Dev", f"{stats['Std Dev']:.2f}")
        with col3:
            st.metric("Min", f"{stats['Min']:.2f}")
        with col4:
            st.metric("Max", f"{stats['Max']:.2f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales Trend")
            fig_ts = eda.plot_time_series(df, date_col, sales_col)
            st.plotly_chart(fig_ts, use_container_width=True)
        
        with col2:
            st.subheader("Sales Distribution")
            fig_dist = eda.plot_distribution(df, sales_col)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Seasonal Pattern
        st.subheader("Monthly Seasonal Pattern")
        fig_seasonal = eda.plot_seasonal_pattern(df, date_col, sales_col, 'Month')
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Trend Analysis
        st.subheader("Trend Analysis")
        fig_trend = eda.analyze_trend(df, date_col, sales_col)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Outliers
        st.subheader("Outlier Detection")
        df_outliers = eda.detect_outliers(df, sales_col)
        outlier_count = df_outliers['Outlier'].sum()
        st.info(f"🔍 Found {outlier_count} outliers ({100*outlier_count/len(df):.2f}% of data)")
    
    else:
        st.warning("⚠️ Please upload and preprocess data first (Tab 1)")

# TAB 3: MODEL TRAINING
with tab3:
    st.markdown("<div class='section-title'>Model Training</div>", unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        df = st.session_state.data
        sales_col = st.selectbox("Select Sales Column", df.columns, key="train_sales")
        
        if st.button("🚀 Train Models", key="train_btn"):
            with st.spinner("Training models... This may take a few minutes"):
                try:
                    # Prepare data
                    X_lstm, y_lstm = st.session_state.preprocessor.prepare_lstm_data(
                        df, sales_col, seq_length=30
                    )
                    X_train, X_test, y_train, y_test = st.session_state.preprocessor.train_test_split_data(
                        X_lstm, y_lstm, test_size=0.2
                    )
                    
                    # Train LSTM
                    st.info("🔨 Training LSTM model...")
                    lstm = LSTMForecaster(epochs=lstm_epochs, batch_size=lstm_batch)
                    lstm.train(X_train, y_train, verbose=0)
                    lstm_preds = lstm.predict(X_test)
                    
                    # Inverse scale LSTM predictions
                    lstm_preds_original = st.session_state.preprocessor.inverse_scale(
                        lstm_preds.reshape(-1, 1)
                    ).flatten()
                    y_test_original = st.session_state.preprocessor.inverse_scale(
                        y_test.reshape(-1, 1)
                    ).flatten()
                    
                    # Train Hybrid Model
                    st.info("🔨 Training Hybrid ARIMA+XGBoost model...")
                    try:
                        p, d, q = map(int, arima_order.split(','))
                        hybrid = HybridArimaXGBoost(arima_order=(p, d, q))
                    except:
                        hybrid = HybridArimaXGBoost()
                    
                    hybrid.fit_arima(df[sales_col].values)
                    hybrid.fit_xgb_on_residuals(df[sales_col].values)
                    
                    # Generate predictions for test set
                    hybrid_preds = []
                    for _ in range(len(y_test)):
                        hybrid_preds.append(hybrid.predict(steps=1)[0])
                    hybrid_preds = np.array(hybrid_preds)
                    
                    # Compare models
                    st.info("📊 Comparing models...")
                    comparison = ModelComparison()
                    results_df = comparison.compare_models(
                        y_test_original,
                        {
                            'LSTM': lstm_preds_original,
                            'Hybrid ARIMA+XGBoost': hybrid_preds
                        }
                    )
                    
                    # Store results
                    st.session_state.predictions = {
                        'lstm': lstm_preds_original,
                        'hybrid': hybrid_preds,
                        'actual': y_test_original,
                        'lstm_model': lstm,
                        'hybrid_model': hybrid,
                        'comparison': results_df
                    }
                    st.session_state.models_trained = True
                    
                    st.success("✅ Models trained successfully!")
                
                except Exception as e:
                    st.error(f"❌ Training error: {str(e)}")
        
        # Display training results if available
        if st.session_state.models_trained:
            st.subheader("📊 Model Comparison Results")
            st.dataframe(st.session_state.predictions['comparison'], use_container_width=True)
            
            # Metrics visualization
            col1, col2, col3 = st.columns(3)
            comparison_df = st.session_state.predictions['comparison']
            
            best_rmse_idx = comparison_df['RMSE'].idxmin()
            best_rmse_model = comparison_df.iloc[best_rmse_idx]
            
            with col1:
                st.metric("Best Model (RMSE)", best_rmse_model['Model'], 
                         delta=f"{best_rmse_model['RMSE']:.2f}")
            with col2:
                st.metric("Best MAE", f"{best_rmse_model['MAE']:.2f}")
            with col3:
                st.metric("Best MAPE", f"{best_rmse_model['MAPE']:.2f}%")
    
    else:
        st.warning("⚠️ Please upload and preprocess data first (Tab 1)")

# TAB 4: FORECASTS
with tab4:
    st.markdown("<div class='section-title'>Demand Forecasts & Comparison</div>", unsafe_allow_html=True)
    
    if st.session_state.models_trained:
        # Plot actual vs predicted
        st.subheader("Test Set: Actual vs Predicted")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.predictions['actual'],
            mode='lines', name='Actual',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            y=st.session_state.predictions['lstm'],
            mode='lines', name='LSTM Prediction',
            line=dict(color='green', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            y=st.session_state.predictions['hybrid'],
            mode='lines', name='Hybrid ARIMA+XGBoost',
            line=dict(color='orange', dash='dash')
        ))
        fig.update_layout(title='Model Predictions vs Actual Values',
                         xaxis_title='Time Period', yaxis_title='Demand',
                         hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Future forecast
        st.subheader("Future Demand Forecast")
        
        if st.button("🔮 Generate Future Forecast", key="forecast_btn"):
            with st.spinner("Generating forecast..."):
                try:
                    # Get last sequence for LSTM
                    last_data = st.session_state.preprocessor.scaler.transform(
                        st.session_state.predictions['actual'][-30:].reshape(-1, 1)
                    ).flatten()
                    
                    # LSTM future forecast
                    lstm_future = st.session_state.predictions['lstm_model'].forecast_future(
                        last_data, steps=forecast_steps,
                        scaler=st.session_state.preprocessor.scaler
                    )
                    
                    # Hybrid future forecast
                    hybrid_future_list = []
                    for _ in range(forecast_steps):
                        hybrid_future_list.append(
                            st.session_state.predictions['hybrid_model'].predict(steps=1)[0]
                        )
                    hybrid_future = np.array(hybrid_future_list)
                    
                    # Plot future forecast
                    future_periods = range(len(st.session_state.predictions['actual']),
                                          len(st.session_state.predictions['actual']) + forecast_steps)
                    
                    fig_future = go.Figure()
                    fig_future.add_trace(go.Scatter(
                        x=range(len(st.session_state.predictions['actual'])),
                        y=st.session_state.predictions['actual'],
                        mode='lines', name='Historical Demand',
                        line=dict(color='blue', width=2)
                    ))
                    fig_future.add_trace(go.Scatter(
                        x=list(future_periods),
                        y=lstm_future,
                        mode='lines+markers', name='LSTM Forecast',
                        line=dict(color='green', dash='dash')
                    ))
                    fig_future.add_trace(go.Scatter(
                        x=list(future_periods),
                        y=hybrid_future,
                        mode='lines+markers', name='Hybrid Forecast',
                        line=dict(color='orange', dash='dash')
                    ))
                    
                    fig_future.update_layout(
                        title=f'Demand Forecast for Next {forecast_steps} Days',
                        xaxis_title='Time Period',
                        yaxis_title='Demand',
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_future, use_container_width=True)
                    
                    # Store forecast
                    st.session_state.forecast_lstm = lstm_future
                    st.session_state.forecast_hybrid = hybrid_future
                    
                    st.success("✅ Forecast generated!")
                
                except Exception as e:
                    st.error(f"❌ Forecast error: {str(e)}")
    
    else:
        st.warning("⚠️ Please train models first (Tab 3)")

# TAB 5: INVENTORY OPTIMIZATION
with tab5:
    st.markdown("<div class='section-title'>Inventory Optimization</div>", unsafe_allow_html=True)
    
    if st.session_state.models_trained and st.session_state.data is not None:
        df = st.session_state.data
        sales_col = st.selectbox("Select Sales Column", df.columns, key="inv_sales")
        
        # Get forecast
        if 'forecast_lstm' not in st.session_state:
            st.warning("⚠️ Please generate forecast first (Tab 4)")
        else:
            try:
                inv_opt = InventoryOptimization(service_level=service_level)
                
                # Use average of LSTM and Hybrid forecasts
                forecast_demand = (st.session_state.forecast_lstm + 
                                 st.session_state.forecast_hybrid) / 2
                
                # Calculate recommendations
                recommendations = inv_opt.generate_inventory_recommendations(
                    df[sales_col].values,
                    lead_time=lead_time,
                    annual_demand=len(df),
                    holding_cost=1.0,
                    ordering_cost=10.0
                )
                
                # Display recommendations
                st.subheader("📦 Inventory Recommendations")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Safety Stock", f"{recommendations['safety_stock']:.0f} units")
                with col2:
                    st.metric("Reorder Point", f"{recommendations['reorder_point']:.0f} units")
                with col3:
                    st.metric("Min Stock", f"{recommendations['min_stock_recommended']:.0f} units")
                with col4:
                    st.metric("Max Stock", f"{recommendations['max_stock_recommended']:.0f} units")
                
                if 'economic_order_quantity' in recommendations:
                    st.metric("EOQ", f"{recommendations['economic_order_quantity']:.0f} units")
                
                # Forecast inventory levels
                st.subheader("Projected Inventory Levels")
                current_stock = df[sales_col].iloc[-1] * 10  # Assume 10 days of stock
                
                inv_forecast = inv_opt.forecast_inventory_levels(
                    current_stock, forecast_demand,
                    recommendations['reorder_point'], lead_time
                )
                
                # Plot inventory forecast
                fig_inv = go.Figure()
                fig_inv.add_trace(go.Scatter(
                    x=inv_forecast['Period'],
                    y=inv_forecast['Inventory_Level'],
                    mode='lines+markers', name='Projected Inventory',
                    fill='tozeroy', line=dict(color='blue')
                ))
                fig_inv.add_hline(
                    y=recommendations['reorder_point'],
                    line_dash="dash", line_color="red",
                    annotation_text="Reorder Point",
                    annotation_position="right"
                )
                fig_inv.add_hline(
                    y=recommendations['safety_stock'],
                    line_dash="dash", line_color="orange",
                    annotation_text="Safety Stock"
                )
                fig_inv.update_layout(
                    title='Projected Inventory Levels',
                    xaxis_title='Forecast Period',
                    yaxis_title='Inventory Level',
                    template='plotly_white'
                )
                st.plotly_chart(fig_inv, use_container_width=True)
                
                # Display detailed report
                st.subheader("Optimization Report")
                report = inv_opt.generate_optimization_report(recommendations)
                st.code(report)
            
            except Exception as e:
                st.error(f"❌ Inventory optimization error: {str(e)}")
    
    else:
        st.warning("⚠️ Please train models and generate forecast first (Tabs 3 & 4)")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 AI-Based Demand Forecasting System v1.0 | Built with TensorFlow, XGBoost, and Streamlit</p>
    <p>LSTM + Hybrid ARIMA+XGBoost for Accurate Demand Prediction and Inventory Optimization</p>
</div>
""", unsafe_allow_html=True)
