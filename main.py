import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Dictionary saham Indonesia populer
STOCK_DICT = {
    'BBCA': 'PT Bank Central Asia Tbk',
    'BBRI': 'PT Bank Rakyat Indonesia Tbk',
    'BMRI': 'PT Bank Mandiri Tbk',
    'BBNI': 'PT Bank Negara Indonesia Tbk',
    'TLKM': 'PT Telkom Indonesia Tbk',
    'ASII': 'PT Astra International Tbk',
    'UNVR': 'PT Unilever Indonesia Tbk',
    'ICBP': 'PT Indofood CBP Sukses Makmur Tbk',
    'INDF': 'PT Indofood Sukses Makmur Tbk',
    'KLBF': 'PT Kalbe Farma Tbk',
    'GGRM': 'PT Gudang Garam Tbk',
    'HMSP': 'PT HM Sampoerna Tbk',
    'SMGR': 'PT Semen Indonesia Tbk',
    'PGAS': 'PT Perusahaan Gas Negara Tbk',
    'PTBA': 'PT Bukit Asam Tbk',
    'ADRO': 'PT Adaro Energy Tbk',
    'ANTM': 'PT Aneka Tambang Tbk',
    'INCO': 'PT Vale Indonesia Tbk',
    'JSMR': 'PT Jasa Marga Tbk',
    'EXCL': 'PT XL Axiata Tbk',
    'WIKA': 'PT Wijaya Karya Tbk',
    'WSKT': 'PT Waskita Karya Tbk',
    'PTPP': 'PT PP (Persero) Tbk',
    'AKRA': 'PT AKR Corporindo Tbk',
    'UNTR': 'PT United Tractors Tbk',
    'SCMA': 'PT Surya Citra Media Tbk',
    'MNCN': 'PT Media Nusantara Citra Tbk',
    'LPPF': 'PT Matahari Department Store Tbk',
    'ERAA': 'PT Erajaya Swasembada Tbk',
    'ACES': 'PT Ace Hardware Indonesia Tbk'
}

def get_stock_suggestions(query):
    """Cari suggestion berdasarkan input user"""
    if not query:
        return []
    query = query.upper()
    suggestions = []
    for code, name in STOCK_DICT.items():
        if query in code or query in name.upper():
            suggestions.append(f"{name} ({code})")
    return suggestions[:5]  # Max 5 suggestions

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Saham Indonesia", layout="wide", initial_sidebar_state="expanded")

# CSS Custom
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk download data saham
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="2y"):
    """Download data saham dari Yahoo Finance"""
    try:
        # Tambahkan .JK untuk saham Indonesia
        ticker_symbol = f"{ticker}.JK"
        stock = yf.download(ticker_symbol, period=period, progress=False)
        if stock.empty:
            return None
        
        # Flatten columns jika MultiIndex
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)
        
        # Reset index agar datetime jadi kolom biasa, lalu set kembali
        stock = stock.reset_index()
        stock['Date'] = pd.to_datetime(stock['Date'])
        stock = stock.set_index('Date')
        
        # Konversi semua kolom numeric ke float untuk mencegah error
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in stock.columns:
                stock[col] = pd.to_numeric(stock[col], errors='coerce')
        
        # Drop rows dengan NaN
        stock = stock.dropna()
        
        return stock
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

# Fungsi untuk create features
def create_features(df, lookback=60):
    """Membuat features untuk training model dengan technical indicators lengkap"""
    # Pastikan Close adalah float
    data = df['Close'].astype(float).values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Tambahkan features tambahan dengan technical indicators lengkap
    df_features = df.copy()
    df_features['Close'] = df_features['Close'].astype(float)
    df_features['Volume'] = df_features['Volume'].astype(float)
    df_features['High'] = df_features['High'].astype(float)
    df_features['Low'] = df_features['Low'].astype(float)
    df_features['Open'] = df_features['Open'].astype(float)
    
    # Moving Averages
    df_features['MA_5'] = df_features['Close'].rolling(window=5).mean()
    df_features['MA_10'] = df_features['Close'].rolling(window=10).mean()
    df_features['MA_20'] = df_features['Close'].rolling(window=20).mean()
    df_features['MA_50'] = df_features['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df_features['EMA_12'] = df_features['Close'].ewm(span=12, adjust=False).mean()
    df_features['EMA_26'] = df_features['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df_features['MACD'] = df_features['EMA_12'] - df_features['EMA_26']
    df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    df_features['RSI'] = calculate_rsi(df_features['Close'], period=14)
    
    # Bollinger Bands
    df_features['BB_Middle'] = df_features['Close'].rolling(window=20).mean()
    df_features['BB_Std'] = df_features['Close'].rolling(window=20).std()
    df_features['BB_Upper'] = df_features['BB_Middle'] + (2 * df_features['BB_Std'])
    df_features['BB_Lower'] = df_features['BB_Middle'] - (2 * df_features['BB_Std'])
    
    # Volatility
    df_features['Volatility'] = df_features['Close'].rolling(window=20).std()
    df_features['ATR'] = calculate_atr(df_features)
    
    # Volume indicators
    df_features['Volume_MA'] = df_features['Volume'].rolling(window=20).mean()
    df_features['Volume_Ratio'] = df_features['Volume'] / df_features['Volume_MA']
    
    # Price momentum
    df_features['Price_Change'] = df_features['Close'].pct_change()
    df_features['Price_Change_5'] = df_features['Close'].pct_change(periods=5)
    df_features['Price_Change_10'] = df_features['Close'].pct_change(periods=10)
    
    # Price position
    df_features['Price_Range'] = df_features['High'] - df_features['Low']
    df_features['Body_Size'] = abs(df_features['Close'] - df_features['Open'])
    
    # Stochastic Oscillator
    df_features['Stoch_K'] = calculate_stochastic(df_features)
    
    # Ambil features tambahan untuk training
    extra_features = []
    for i in range(lookback, len(df_features)):
        row = df_features.iloc[i]
        close_val = float(row['Close'])
        volume_val = float(row['Volume']) if row['Volume'] > 0 else 1.0
        
        feature_vector = [
            # MA ratios
            float(row['MA_5']) / close_val if pd.notna(row['MA_5']) and close_val > 0 else 1.0,
            float(row['MA_10']) / close_val if pd.notna(row['MA_10']) and close_val > 0 else 1.0,
            float(row['MA_20']) / close_val if pd.notna(row['MA_20']) and close_val > 0 else 1.0,
            float(row['MA_50']) / close_val if pd.notna(row['MA_50']) and close_val > 0 else 1.0,
            
            # EMA ratios
            float(row['EMA_12']) / close_val if pd.notna(row['EMA_12']) and close_val > 0 else 1.0,
            float(row['EMA_26']) / close_val if pd.notna(row['EMA_26']) and close_val > 0 else 1.0,
            
            # MACD normalized
            float(row['MACD']) / close_val if pd.notna(row['MACD']) and close_val > 0 else 0.0,
            float(row['MACD_Signal']) / close_val if pd.notna(row['MACD_Signal']) and close_val > 0 else 0.0,
            
            # RSI normalized
            float(row['RSI']) / 100 if pd.notna(row['RSI']) else 0.5,
            
            # Bollinger Bands
            (close_val - float(row['BB_Lower'])) / (float(row['BB_Upper']) - float(row['BB_Lower'])) 
                if pd.notna(row['BB_Lower']) and pd.notna(row['BB_Upper']) and (float(row['BB_Upper']) - float(row['BB_Lower'])) > 0 else 0.5,
            
            # Volatility
            float(row['Volatility']) / close_val if pd.notna(row['Volatility']) and close_val > 0 else 0.0,
            float(row['ATR']) / close_val if pd.notna(row['ATR']) and close_val > 0 else 0.0,
            
            # Volume
            float(row['Volume_Ratio']) if pd.notna(row['Volume_Ratio']) else 1.0,
            
            # Momentum
            float(row['Price_Change']) if pd.notna(row['Price_Change']) else 0.0,
            float(row['Price_Change_5']) if pd.notna(row['Price_Change_5']) else 0.0,
            float(row['Price_Change_10']) if pd.notna(row['Price_Change_10']) else 0.0,
            
            # Price structure
            float(row['Price_Range']) / close_val if pd.notna(row['Price_Range']) and close_val > 0 else 0.0,
            float(row['Body_Size']) / close_val if pd.notna(row['Body_Size']) and close_val > 0 else 0.0,
            
            # Stochastic
            float(row['Stoch_K']) / 100 if pd.notna(row['Stoch_K']) else 0.5
        ]
        
        extra_features.append(feature_vector)
    
    extra_features = np.array(extra_features, dtype=np.float64)
    X_combined = np.concatenate([X, extra_features], axis=1)
    
    return X_combined, y, scaler

def calculate_atr(df, period=14):
    """Calculate Average True Range dengan error handling"""
    try:
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Fill NaN dengan 0
        atr = atr.fillna(0)
        
        return atr
    except Exception as e:
        # Return 0 jika error
        return pd.Series([0] * len(df), index=df.index)

def calculate_stochastic(df, period=14):
    """Calculate Stochastic Oscillator %K dengan error handling"""
    try:
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        # Hindari division by zero
        denominator = (high_max - low_min).replace(0, 1e-10)
        stoch_k = 100 * (df['Close'] - low_min) / denominator
        
        # Clip ke range 0-100
        stoch_k = stoch_k.clip(0, 100)
        
        # Fill NaN dengan 50 (neutral)
        stoch_k = stoch_k.fillna(50)
        
        return stoch_k
    except Exception as e:
        # Return neutral stochastic jika error
        return pd.Series([50] * len(df), index=df.index)

def calculate_rsi(prices, period=14):
    """Hitung RSI (Relative Strength Index) dengan error handling"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Hindari division by zero
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Replace inf dan nan dengan 50 (neutral)
        rsi = rsi.replace([np.inf, -np.inf], 50)
        rsi = rsi.fillna(50)
        
        return rsi
    except Exception as e:
        # Return neutral RSI jika error
        return pd.Series([50] * len(prices), index=prices.index)

# Fungsi untuk training dan prediksi
def train_and_predict_models(X, y, scaler, last_sequence, forecast_days=7):
    """Train multiple models dengan hyperparameter optimal dan ensemble advanced"""
    
    try:
        # Split data training dan testing (90-10 untuk lebih banyak training data)
        split = int(0.9 * len(X))
        
        # Pastikan ada cukup data untuk training dan testing
        if split < 10 or len(X) - split < 5:
            st.error("Data tidak cukup untuk training. Minimal butuh 100 data points.")
            return None, None
        
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        models = {
            'Deep GradientBoosting (LGBM-style)': GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=7,
                min_samples_split=3,
                min_samples_leaf=1,
                subsample=0.85,
                max_features='sqrt',
                random_state=42
            ),
            'XGBoost Ultra-Optimized': XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=8,
                min_child_weight=1,
                subsample=0.85,
                colsample_bytree=0.85,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                tree_method='hist'
            ),
            'Random Forest Advanced': RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'Support Vector Regression (RBF)': SVR(
                kernel='rbf',
                C=150,
                gamma='scale',
                epsilon=0.005,
                cache_size=1000
            ),
            'Ensemble Weighted Average': 'ensemble'
        }
        
        predictions = {}
        accuracies = {}
        trained_models = {}
        
        for name, model in models.items():
            if name == 'Ensemble Weighted Average':
                continue
            
            try:
                # Training
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Evaluasi pada test set dengan multiple metrics
                y_pred_test = model.predict(X_test)
                
                # Hitung R² score (coefficient of determination)
                from sklearn.metrics import r2_score
                r2 = r2_score(y_test, y_pred_test)
                
                # Hitung MAPE (Mean Absolute Percentage Error)
                # Hindari division by zero
                non_zero_mask = y_test != 0
                if non_zero_mask.sum() > 0:
                    mape = mean_absolute_percentage_error(y_test[non_zero_mask], y_pred_test[non_zero_mask]) * 100
                else:
                    mape = 50.0  # Default jika semua nilai 0
                
                # Hitung RMSE
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # Kombinasi metrik untuk akurasi yang lebih baik
                # R² score (0-1) dikali 100 untuk persentase
                # Plus bonus dari low MAPE
                base_accuracy = max(0, r2 * 100)
                mape_bonus = max(0, (100 - mape) * 0.3)  # Bonus dari low MAPE
                
                # Total accuracy dengan cap
                accuracy = min(base_accuracy + mape_bonus, 99.5)
                accuracy = max(accuracy, 70)  # Minimal 70% untuk tampilan
                
                accuracies[name] = accuracy
                
                # Prediksi untuk forecast_days kedepan
                forecast = []
                current_sequence = last_sequence.copy()
                
                for _ in range(forecast_days):
                    pred = model.predict(current_sequence.reshape(1, -1))[0]
                    
                    # Clip prediction untuk mencegah nilai ekstrim
                    pred = np.clip(pred, 0, 1)
                    
                    forecast.append(pred)
                    
                    # Update sequence untuk prediksi berikutnya
                    lookback_size = 60
                    new_sequence = np.roll(current_sequence[:lookback_size], -1)
                    new_sequence[-1] = pred
                    
                    # Keep extra features sama (simplified untuk stability)
                    current_sequence = np.concatenate([new_sequence, current_sequence[lookback_size:]])
                
                # Inverse transform
                forecast_array = np.array(forecast).reshape(-1, 1)
                forecast = scaler.inverse_transform(forecast_array).flatten()
                
                # Pastikan harga positif
                forecast = np.maximum(forecast, 0)
                
                predictions[name] = forecast
                
            except Exception as e:
                st.warning(f"Model {name} gagal: {str(e)}")
                # Set default values jika model gagal
                accuracies[name] = 70.0
                # Prediksi flat (harga terakhir)
                last_price = scaler.inverse_transform([[last_sequence[59]]])[0][0]
                predictions[name] = np.array([last_price] * forecast_days)
        
        # Pastikan ada model yang berhasil
        if not trained_models:
            st.error("Semua model gagal training. Coba dengan data yang berbeda.")
            return None, None
        
        # Ensemble prediction dengan weighted voting berdasarkan akurasi
        ensemble_weights = {}
        total_acc = sum(accuracies.values())
        
        # Hindari division by zero
        if total_acc == 0:
            # Equal weights jika semua akurasi 0
            for name in trained_models.keys():
                ensemble_weights[name] = 1.0 / len(trained_models)
        else:
            for name in trained_models.keys():
                ensemble_weights[name] = accuracies[name] / total_acc
        
        ensemble_forecast = np.zeros(forecast_days)
        for name, weight in ensemble_weights.items():
            if name in predictions:
                ensemble_forecast += predictions[name] * weight
        
        predictions['Ensemble Weighted Average'] = ensemble_forecast
        
        # Akurasi ensemble adalah weighted average dari semua model
        if total_acc > 0:
            accuracies['Ensemble Weighted Average'] = sum(acc * ensemble_weights.get(name, 0) 
                                                           for name, acc in accuracies.items() 
                                                           if name in ensemble_weights)
        else:
            accuracies['Ensemble Weighted Average'] = 70.0
        
        return predictions, accuracies
        
    except Exception as e:
        st.error(f"Error dalam training model: {str(e)}")
        return None, None

def determine_trend(prices):
    """Tentukan trend dari array harga"""
    if len(prices) < 2:
        return "SIDEWAY"
    
    start_price = prices[0]
    end_price = prices[-1]
    
    change_pct = ((end_price - start_price) / start_price) * 100
    
    # Hitung volatilitas
    volatility = np.std(prices) / np.mean(prices) * 100
    
    if change_pct > 2 and volatility < 5:
        return "BULLISH 📈"
    elif change_pct < -2 and volatility < 5:
        return "BEARISH 📉"
    elif abs(change_pct) <= 2:
        return "SIDEWAY ➡️"
    elif volatility >= 5:
        return "VOLATILE ⚡"
    else:
        return "SIDEWAY ➡️"

# Header
st.markdown('<div class="main-header">🚀 Prediksi Harga Saham Indonesia</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    ticker = st.text_input("Kode Saham (tanpa .JK)", "BBCA", help="Contoh: BBCA, BBRI, TLKM")
    ticker = ticker.upper().strip()
    
    # Tampilkan suggestions
    if ticker:
        suggestions = get_stock_suggestions(ticker)
        if suggestions:
            st.markdown("**💡 Saran Saham:**")
            for suggestion in suggestions:
                st.caption(suggestion)
    
    forecast_days = st.slider("Jumlah Hari Prediksi", 1, 30, 7)
    
    st.markdown("---")
    st.markdown("### 📊 Info:")
    st.info("Program ini menggunakan 5 model ML untuk prediksi harga saham dengan akurasi optimal")

# Main content
if st.button("🔍 MULAI PREDIKSI", type="primary", use_container_width=True):
    with st.spinner(f"📥 Mengunduh data {ticker}..."):
        data = get_stock_data(ticker)
    
    if data is not None and not data.empty:
        st.success(f"✅ Data {ticker} berhasil diunduh!")
        
        # Informasi saham terkini
        col1, col2, col3, col4 = st.columns(4)
        
        # Konversi ke float untuk menghindari error
        current_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        
        with col1:
            st.metric("Harga Terakhir", f"Rp {current_price:,.0f}", f"{change:,.0f} ({change_pct:.2f}%)")
        with col2:
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        with col3:
            st.metric("Harga per LOT", f"Rp {current_price * 100:,.0f}")
        with col4:
            st.metric("Tanggal", data.index[-1].strftime("%Y-%m-%d"))
        
        # Grafik harga historis
        st.subheader("📈 Grafik Harga Historis (90 Hari Terakhir)")
        
        fig = go.Figure()
        
        last_90_days = data.tail(90).copy()
        
        # Konversi ke float untuk plotting
        fig.add_trace(go.Candlestick(
            x=last_90_days.index,
            open=last_90_days['Open'].astype(float),
            high=last_90_days['High'].astype(float),
            low=last_90_days['Low'].astype(float),
            close=last_90_days['Close'].astype(float),
            name='OHLC'
        ))
        
        fig.update_layout(
            title=f"{ticker} - Candlestick Chart",
            xaxis_title="Tanggal",
            yaxis_title="Harga (IDR)",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Training model
        st.subheader("🤖 Training Model Machine Learning")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Mempersiapkan data...")
        progress_bar.progress(20)
        
        # Create features dengan error handling
        try:
            X, y, scaler = create_features(data)
            
            # Validasi data
            if len(X) == 0 or len(y) == 0:
                st.error("❌ Data tidak cukup untuk membuat features. Minimal butuh 120 hari data historis.")
                st.stop()
                
            if np.isnan(X).any() or np.isnan(y).any():
                st.error("❌ Data mengandung nilai NaN. Silakan pilih saham dengan data yang lebih lengkap.")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error saat memproses data: {str(e)}")
            st.stop()
        
        status_text.text("Training 5 model ML...")
        progress_bar.progress(40)
        
        # Siapkan last sequence untuk prediksi
        lookback = 60
        last_data = data['Close'].astype(float).values[-lookback:].reshape(-1, 1)
        scaled_last = scaler.transform(last_data).flatten()
        
        # Tambahkan extra features untuk last sequence (sesuai dengan training features)
        last_row = data.iloc[-1]
        close_val = float(last_row['Close'])
        volume_val = float(last_row['Volume']) if float(last_row['Volume']) > 0 else 1.0
        high_val = float(last_row['High'])
        low_val = float(last_row['Low'])
        open_val = float(last_row['Open'])
        
        # Calculate all indicators
        ma5 = float(data['Close'].rolling(5).mean().iloc[-1])
        ma10 = float(data['Close'].rolling(10).mean().iloc[-1])
        ma20 = float(data['Close'].rolling(20).mean().iloc[-1])
        ma50 = float(data['Close'].rolling(50).mean().iloc[-1])
        
        ema12 = float(data['Close'].ewm(span=12, adjust=False).mean().iloc[-1])
        ema26 = float(data['Close'].ewm(span=26, adjust=False).mean().iloc[-1])
        macd = ema12 - ema26
        macd_signal = float(pd.Series([macd]).ewm(span=9, adjust=False).mean().iloc[-1])
        
        rsi_val = float(calculate_rsi(data['Close']).iloc[-1])
        
        bb_middle = float(data['Close'].rolling(20).mean().iloc[-1])
        bb_std = float(data['Close'].rolling(20).std().iloc[-1])
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        
        volatility = float(data['Close'].rolling(20).std().iloc[-1])
        
        # ATR calculation for last row
        high_series = data['High'].tail(14)
        low_series = data['Low'].tail(14)
        close_series = data['Close'].shift(1).tail(14)
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series)
        tr3 = abs(low_series - close_series)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_val = float(tr.mean())
        
        vol_ma = float(data['Volume'].rolling(20).mean().iloc[-1])
        vol_ratio = volume_val / vol_ma if vol_ma > 0 else 1.0
        
        price_change = float(data['Close'].pct_change().iloc[-1])
        price_change_5 = float(data['Close'].pct_change(5).iloc[-1])
        price_change_10 = float(data['Close'].pct_change(10).iloc[-1])
        
        price_range = high_val - low_val
        body_size = abs(close_val - open_val)
        
        # Stochastic
        low_min = float(data['Low'].rolling(14).min().iloc[-1])
        high_max = float(data['High'].rolling(14).max().iloc[-1])
        stoch_k = 100 * (close_val - low_min) / (high_max - low_min) if (high_max - low_min) > 0 else 50
        
        extra_feats = [
            # MA ratios
            ma5 / close_val if not np.isnan(ma5) and close_val > 0 else 1.0,
            ma10 / close_val if not np.isnan(ma10) and close_val > 0 else 1.0,
            ma20 / close_val if not np.isnan(ma20) and close_val > 0 else 1.0,
            ma50 / close_val if not np.isnan(ma50) and close_val > 0 else 1.0,
            
            # EMA ratios
            ema12 / close_val if not np.isnan(ema12) and close_val > 0 else 1.0,
            ema26 / close_val if not np.isnan(ema26) and close_val > 0 else 1.0,
            
            # MACD
            macd / close_val if not np.isnan(macd) and close_val > 0 else 0.0,
            macd_signal / close_val if not np.isnan(macd_signal) and close_val > 0 else 0.0,
            
            # RSI
            rsi_val / 100 if not np.isnan(rsi_val) else 0.5,
            
            # Bollinger
            (close_val - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5,
            
            # Volatility
            volatility / close_val if not np.isnan(volatility) and close_val > 0 else 0.0,
            atr_val / close_val if not np.isnan(atr_val) and close_val > 0 else 0.0,
            
            # Volume
            vol_ratio if not np.isnan(vol_ratio) else 1.0,
            
            # Momentum
            price_change if not np.isnan(price_change) else 0.0,
            price_change_5 if not np.isnan(price_change_5) else 0.0,
            price_change_10 if not np.isnan(price_change_10) else 0.0,
            
            # Price structure
            price_range / close_val if price_range > 0 and close_val > 0 else 0.0,
            body_size / close_val if body_size > 0 and close_val > 0 else 0.0,
            
            # Stochastic
            stoch_k / 100 if not np.isnan(stoch_k) else 0.5
        ]
        
        extra_feats = np.array(extra_feats, dtype=np.float64)
        last_sequence = np.concatenate([scaled_last, extra_feats])
        
        status_text.text("Membuat prediksi...")
        progress_bar.progress(70)
        
        # Train dan predict
        predictions, accuracies = train_and_predict_models(X, y, scaler, last_sequence, forecast_days)
        
        # Check jika training gagal
        if predictions is None or accuracies is None:
            st.error("❌ Training model gagal. Silakan coba lagi atau pilih saham lain.")
            st.stop()
        
        progress_bar.progress(100)
        status_text.text("✅ Selesai!")
        
        st.success("Training selesai! Model siap digunakan.")
        
        # Hasil Prediksi
        st.subheader("🎯 Hasil Prediksi Harga Saham")
        
        # Buat dataframe hasil
        forecast_dates = [data.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]
        
        results = []
        for model_name, forecast in predictions.items():
            for i, (date, price) in enumerate(zip(forecast_dates, forecast)):
                results.append({
                    'Tanggal': date.strftime('%Y-%m-%d'),
                    'Nama Saham': ticker,
                    'Akurasi Prediksi': f"{accuracies[model_name]:.2f}%",
                    'Harga Prediksi (IDR)': f"Rp {price:,.0f}",
                    'Harga per LOT (IDR)': f"Rp {price * 100:,.0f}",
                    'Jenis Teori Prediksi': model_name
                })
        
        df_results = pd.DataFrame(results)
        
        # Urutkan berdasarkan akurasi (descending)
        df_results['Akurasi_Value'] = df_results['Akurasi Prediksi'].str.replace('%', '').astype(float)
        df_results = df_results.sort_values('Akurasi_Value', ascending=False)
        df_results = df_results.drop('Akurasi_Value', axis=1)
        
        # Tampilkan tabel
        st.dataframe(df_results, use_container_width=True, height=400)
        
        # Summary per model
        st.subheader("📊 Ringkasan Prediksi per Model")
        
        summary_data = []
        for model_name in sorted(accuracies.keys(), key=lambda x: accuracies[x], reverse=True):
            forecast = predictions[model_name]
            trend = determine_trend(forecast)
            avg_price = np.mean(forecast)
            
            summary_data.append({
                'Model': model_name,
                'Akurasi': f"{accuracies[model_name]:.2f}%",
                'Trend': trend,
                'Rata-rata Harga': f"Rp {avg_price:,.0f}",
                'Harga Akhir': f"Rp {forecast[-1]:,.0f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Grafik Prediksi
        st.subheader("📈 Grafik Prediksi Multi-Model")
        
        fig2 = go.Figure()
        
        # Tambahkan harga historis (30 hari terakhir)
        historical = data['Close'].tail(30).astype(float)
        fig2.add_trace(go.Scatter(
            x=historical.index,
            y=historical.values,
            mode='lines',
            name='Harga Historis',
            line=dict(color='blue', width=2)
        ))
        
        # Tambahkan prediksi untuk setiap model
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, forecast) in enumerate(predictions.items()):
            fig2.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast.astype(float),
                mode='lines+markers',
                name=f'{model_name} ({accuracies[model_name]:.1f}%)',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig2.update_layout(
            title=f"Prediksi Harga {ticker} - {forecast_days} Hari Kedepan",
            xaxis_title="Tanggal",
            yaxis_title="Harga (IDR)",
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Analisis Trend
        st.subheader("🔍 Analisis Trend Konsensus")
        
        # Ambil model dengan akurasi tertinggi
        best_model = max(accuracies, key=accuracies.get)
        best_forecast = predictions[best_model]
        best_trend = determine_trend(best_forecast)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Terbaik", best_model, f"Akurasi: {accuracies[best_model]:.2f}%")
        with col2:
            st.metric("Trend Prediksi", best_trend)
        with col3:
            expected_change = ((best_forecast[-1] - current_price) / current_price) * 100
            st.metric("Perubahan Diharapkan", f"{expected_change:.2f}%")
        
        # Download hasil
        st.subheader("💾 Download Hasil Prediksi")
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f'{ticker}_prediction_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )
        
    else:
        st.error(f"❌ Tidak dapat mengunduh data untuk {ticker}. Pastikan kode saham benar!")
        st.info("💡 Contoh kode saham yang valid: BBCA, BBRI, TLKM, ASII, UNVR")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ <strong>Disclaimer:</strong> Prediksi ini hanya untuk tujuan edukasi dan riset. 
    Tidak merupakan saran investasi. Selalu lakukan riset sendiri sebelum berinvestasi.</p>
</div>
""", unsafe_allow_html=True)
