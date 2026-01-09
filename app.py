import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Housing Price Prediction App",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏠 Housing Price Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("This interactive dashboard allows you to predict house prices and explore housing market insights.")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Housing.csv")

data = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("📊 Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Mode",
    ["📈 Data Explorer", "🔍 Price Predictor", "📊 Market Insights", "⚙️ Model Analysis"]
)

# --------------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------------
df = data.copy()

# Encode categorical variables
categorical_cols = df.select_dtypes(include=["object"]).columns
encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# Features & Target
X = df.drop("price", axis=1)
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
@st.cache_resource
def train_models():
    models = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models['Linear Regression'] = lr
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    models['Ridge Regression'] = ridge
    
    # Lasso Regression
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)
    models['Lasso Regression'] = lasso
    
    return models

models = train_models()

# --------------------------------------------------
# MODE 1: DATA EXPLORER
# --------------------------------------------------
if app_mode == "📈 Data Explorer":
    st.subheader("📊 Dataset Overview")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Features", len(data.columns) - 1)
    with col3:
        st.metric("Avg Price", f"KES {data['price'].mean():,.0f}")
    with col4:
        st.metric("Avg Area", f"{data['area'].mean():.0f} sq ft")
    
    # Dataset preview with filtering
    st.subheader("Dataset Preview")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        preview_rows = st.slider("Rows to display", 5, 50, 10)
    with col2:
        show_raw_data = st.checkbox("Show raw data", True)
    
    if show_raw_data:
        st.dataframe(data.head(preview_rows), use_container_width=True)
    
    # Data information
    with st.expander("📋 Dataset Information"):
        st.write("**Columns:**", list(data.columns))
        st.write("**Data Types:**")
        st.write(data.dtypes)
        st.write("**Missing Values:**")
        st.write(data.isnull().sum())
    
    # Interactive filtering
    st.subheader("🔍 Interactive Filtering")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        min_price = st.slider("Min Price (KES)", 
                            int(data['price'].min()), 
                            int(data['price'].max()), 
                            int(data['price'].min()))
        max_price = st.slider("Max Price (KES)", 
                            int(data['price'].min()), 
                            int(data['price'].max()), 
                            int(data['price'].max()))
    
    with filter_col2:
        min_area = st.slider("Min Area (sq ft)", 
                           int(data['area'].min()), 
                           int(data['area'].max()), 
                           int(data['area'].min()))
        max_area = st.slider("Max Area (sq ft)", 
                           int(data['area'].min()), 
                           int(data['area'].max()), 
                           int(data['area'].max()))
    
    with filter_col3:
        bedrooms_filter = st.multiselect("Bedrooms", 
                                       sorted(data['bedrooms'].unique()),
                                       default=sorted(data['bedrooms'].unique()))
    
    filtered_data = data[
        (data['price'] >= min_price) & 
        (data['price'] <= max_price) &
        (data['area'] >= min_area) & 
        (data['area'] <= max_area) &
        (data['bedrooms'].isin(bedrooms_filter))
    ]
    
    st.metric("Filtered Records", len(filtered_data))
    st.dataframe(filtered_data.head(20), use_container_width=True)

# --------------------------------------------------
# MODE 2: PRICE PREDICTOR
# --------------------------------------------------
elif app_mode == "🔍 Price Predictor":
    st.subheader("🏡 Predict House Price")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Property Specifications
        st.markdown("### Property Specifications")
        
        col1a, col1b = st.columns(2)
        with col1a:
            area = st.number_input("Area (sq ft)", 
                                 min_value=100, 
                                 max_value=50000, 
                                 value=3000,
                                 step=100)
            bedrooms = st.number_input("Bedrooms", 
                                     min_value=1, 
                                     max_value=10, 
                                     value=3)
            bathrooms = st.number_input("Bathrooms", 
                                      min_value=1, 
                                      max_value=10, 
                                      value=2)
            stories = st.number_input("Stories", 
                                    min_value=1, 
                                    max_value=5, 
                                    value=2)
        
        with col1b:
            parking = st.number_input("Parking Spaces", 
                                    min_value=0, 
                                    max_value=10, 
                                    value=1)
            mainroad = st.selectbox("Main Road Access", ["yes", "no"])
            guestroom = st.selectbox("Guest Room", ["yes", "no"])
            basement = st.selectbox("Basement", ["yes", "no"])
        
        col2a, col2b = st.columns(2)
        with col2a:
            hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
            airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
        
        with col2b:
            prefarea = st.selectbox("Preferred Area", ["yes", "no"])
            furnishingstatus = st.selectbox(
                "Furnishing Status",
                ["furnished", "semi-furnished", "unfurnished"]
            )
    
    with col2:
        # Quick Insights Panel
        st.markdown("### 📊 Quick Insights")
        
        # Calculate property score
        property_score = 0
        if area > data['area'].median():
            property_score += 1
        if bedrooms > data['bedrooms'].median():
            property_score += 1
        if bathrooms > data['bathrooms'].median():
            property_score += 1
        if airconditioning == "yes":
            property_score += 1
        if prefarea == "yes":
            property_score += 1
        
        st.markdown(f'<div class="metric-card">Property Score: {property_score}/5</div>', unsafe_allow_html=True)
        
        # Market comparison
        avg_price_per_sqft = data['price'].mean() / data['area'].mean()
        estimated_base = area * avg_price_per_sqft
        st.markdown(f'<div class="metric-card">Avg Rate: KES {avg_price_per_sqft:,.0f}/sq ft</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">Base Estimate: KES {estimated_base:,.0f}</div>', unsafe_allow_html=True)
    
    # Model selection
    st.markdown("### Model Selection")
    model_choice = st.selectbox("Choose Prediction Model", 
                              list(models.keys()),
                              index=0)
    
    # Encode inputs
    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": encoders['mainroad'].transform([mainroad])[0],
        "guestroom": encoders['guestroom'].transform([guestroom])[0],
        "basement": encoders['basement'].transform([basement])[0],
        "hotwaterheating": encoders['hotwaterheating'].transform([hotwaterheating])[0],
        "airconditioning": encoders['airconditioning'].transform([airconditioning])[0],
        "parking": parking,
        "prefarea": encoders['prefarea'].transform([prefarea])[0],
        "furnishingstatus": encoders['furnishingstatus'].transform([furnishingstatus])[0]
    }
    
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    
    # Predict button
    if st.button("🚀 Predict Price", use_container_width=True):
        selected_model = models[model_choice]
        prediction = selected_model.predict(input_scaled)[0]
        
        # Confidence interval (simplified)
        mae = mean_absolute_error(y_test, selected_model.predict(X_test_scaled))
        lower_bound = prediction - 1.96 * mae
        upper_bound = prediction + 1.96 * mae
        
        # Display results
        st.markdown("---")
        col_result1, col_result2, col_result3 = st.columns([2, 1, 1])
        
        with col_result1:
            st.markdown(f'### 💰 Predicted Price: **KES {prediction:,.0f}**')
            st.metric("Price Range", f"KES {lower_bound:,.0f} - {upper_bound:,.0f}")
        
        with col_result2:
            st.metric("Price per sq ft", f"KES {prediction/area:,.0f}")
        
        with col_result3:
            price_percentile = (data['price'] <= prediction).mean() * 100
            st.metric("Market Percentile", f"{price_percentile:.1f}%")
        
        # Feature importance visualization
        st.markdown("### 📊 Feature Impact on Price")
        
        # Get coefficients for linear regression
        if model_choice == "Linear Regression":
            coefs = selected_model.coef_
            features = X.columns
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': features,
                'Coefficient': coefs,
                'Abs_Coefficient': np.abs(coefs)
            }).sort_values('Abs_Coefficient', ascending=True)
            
            fig = px.bar(importance_df, 
                        x='Coefficient', 
                        y='Feature',
                        orientation='h',
                        title='Feature Importance (Linear Regression Coefficients)',
                        color='Coefficient',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# MODE 3: MARKET INSIGHTS
# --------------------------------------------------
elif app_mode == "📊 Market Insights":
    st.subheader("📈 Housing Market Insights")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Distribution", "🏠 Property Features", "💰 Price Correlations", "📍 Geographical Patterns"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution histogram
            fig1 = px.histogram(data, 
                              x='price',
                              nbins=50,
                              title='Price Distribution',
                              labels={'price': 'Price (KES)'},
                              color_discrete_sequence=['#3B82F6'])
            fig1.update_layout(bargap=0.1)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Price by furnishing status
            fig2 = px.box(data, 
                         x='furnishingstatus',
                         y='price',
                         title='Price by Furnishing Status',
                         labels={'furnishingstatus': 'Furnishing Status', 'price': 'Price (KES)'},
                         color='furnishingstatus')
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Bedrooms vs Price
            fig3 = px.box(data,
                         x='bedrooms',
                         y='price',
                         title='Price by Number of Bedrooms',
                         labels={'bedrooms': 'Bedrooms', 'price': 'Price (KES)'})
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Area vs Price scatter
            fig4 = px.scatter(data,
                            x='area',
                            y='price',
                            color='airconditioning',
                            title='Area vs Price',
                            labels={'area': 'Area (sq ft)', 'price': 'Price (KES)'},
                            trendline='ols')
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        # Correlation heatmap
        numeric_data = data.copy()
        for col in categorical_cols:
            numeric_data[col] = encoders[col].transform(numeric_data[col])
        
        correlation_matrix = numeric_data.corr()
        
        fig5 = px.imshow(correlation_matrix,
                        text_auto='.2f',
                        aspect='auto',
                        color_continuous_scale='RdBu',
                        title='Feature Correlation Matrix')
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab4:
        # Create hypothetical coordinates for visualization
        np.random.seed(42)
        data['lat'] = np.random.uniform(-1.5, 1.5, len(data))
        data['lon'] = np.random.uniform(34.5, 37.5, len(data))
        
        fig6 = px.scatter_mapbox(data,
                                lat='lat',
                                lon='lon',
                                color='price',
                                size='area',
                                hover_name='price',
                                hover_data=['bedrooms', 'bathrooms', 'area'],
                                color_continuous_scale='Viridis',
                                zoom=5,
                                title='Price Distribution (Simulated Locations)',
                                mapbox_style='carto-positron')
        st.plotly_chart(fig6, use_container_width=True)

# --------------------------------------------------
# MODE 4: MODEL ANALYSIS
# --------------------------------------------------
elif app_mode == "⚙️ Model Analysis":
    st.subheader("🤖 Model Performance Analysis")
    
    # Model comparison
    st.markdown("### Model Performance Metrics")
    
    performance_data = []
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        performance_data.append({
            'Model': name,
            'MAE': f"KES {mae:,.0f}",
            'RMSE': f"KES {rmse:,.0f}",
            'R² Score': f"{r2:.4f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    st.table(performance_df)
    
    # Actual vs Predicted plot
    st.markdown("### Actual vs Predicted Prices")
    
    model_for_plot = st.selectbox("Select model for visualization", list(models.keys()))
    
    selected_model = models[model_for_plot]
    y_pred = selected_model.predict(X_test_scaled)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Error': y_test.values - y_pred
    })
    
    # Scatter plot
    fig7 = px.scatter(comparison_df,
                     x='Actual',
                     y='Predicted',
                     title=f'Actual vs Predicted Prices ({model_for_plot})',
                     labels={'Actual': 'Actual Price (KES)', 'Predicted': 'Predicted Price (KES)'},
                     trendline='ols',
                     hover_data=['Error'])
    
    # Add perfect prediction line
    max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
    min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
    fig7.add_trace(go.Scatter(x=[min_val, max_val], 
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')))
    
    st.plotly_chart(fig7, use_container_width=True)
    
    # Error distribution
    fig8 = px.histogram(comparison_df,
                       x='Error',
                       nbins=50,
                       title='Prediction Error Distribution',
                       labels={'Error': 'Prediction Error (KES)'})
    fig8.add_vline(x=0, line_dash='dash', line_color='red')
    st.plotly_chart(fig8, use_container_width=True)
    
    # Feature importance for all models
    st.markdown("### Feature Importance Comparison")
    
    importance_fig = make_subplots(rows=1, cols=len(models), 
                                 subplot_titles=list(models.keys()),
                                 shared_yaxes=True)
    
    for idx, (name, model) in enumerate(models.items(), 1):
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            features = X.columns
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Coefficient': coefs
            }).sort_values('Coefficient')
            
            importance_fig.add_trace(
                go.Bar(x=importance_df['Coefficient'],
                      y=importance_df['Feature'],
                      orientation='h',
                      name=name),
                row=1, col=idx
            )
    
    importance_fig.update_layout(height=600, 
                               showlegend=False,
                               title_text="Feature Coefficients Across Models")
    st.plotly_chart(importance_fig, use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
with col_f2:
    st.caption("🏠 Housing Price Prediction Dashboard | Made with Streamlit & Scikit-learn")
    st.caption("📊 Regression Analysis | Predictive Modeling | Market Insights")

# Download predictions button
if app_mode == "🔍 Price Predictor":
    if st.button("💾 Save Prediction Report"):
        report = f"""
        HOUSING PRICE PREDICTION REPORT
        ================================
        
        Property Specifications:
        - Area: {area} sq ft
        - Bedrooms: {bedrooms}
        - Bathrooms: {bathrooms}
        - Stories: {stories}
        - Parking Spaces: {parking}
        - Main Road Access: {mainroad}
        - Guest Room: {guestroom}
        - Basement: {basement}
        - Hot Water Heating: {hotwaterheating}
        - Air Conditioning: {airconditioning}
        - Preferred Area: {prefarea}
        - Furnishing Status: {furnishingstatus}
        
        Prediction:
        - Model Used: {model_choice}
        - Predicted Price: KES {models[model_choice].predict(input_scaled)[0]:,.0f}
        - Price per sq ft: KES {(models[model_choice].predict(input_scaled)[0]/area):,.0f}
        
        Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"housing_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

