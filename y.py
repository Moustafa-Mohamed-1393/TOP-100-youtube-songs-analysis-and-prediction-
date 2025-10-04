# youtube_full_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Try optional libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# -------------------------
# Streamlit must-call first
# -------------------------
st.set_page_config(
    page_title="YouTube Full Analytics & Predictive Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Helper / sample data
# -------------------------
@st.cache_data
def sample_df():
    data = {
        'title': [
            "ROSÉ & Bruno Mars - APT. (Official Music Video)",
            "Lady Gaga, Bruno Mars - Die With A Smile (Official Music Video)",
            "Reneé Rapp - Leave Me Alone (Official Music Video)",
            "Billie Eilish - BIRDS OF A FEATHER (Official Music Video)",
            "Reneé Rapp - Mad (Official Music Video)",
            "Sabrina Carpenter - Espresso",
            "Lady Gaga - Abracadabra (Official Music Video)",
            "Ed Sheeran - Sapphire (Official Music Video)",
            "yung kai - blue (official music video)",
            "Billie Eilish - WILDFLOWER (BILLIE BY FINNEAS)"
        ],
        'description': ["High production"]*10,
        'view_count': [2009014557, 1324833300, 2536628, 558329099, 2113548, 472570966, 191073418, 184696317, 187281056, 40408980],
        'tags': ["tag1;tag2"]*10,
        'duration': [173, 252, 160, 231, 180, 201, 269, 183, 221, 261],
        'duration_string': ["2:53","4:12","2:40","3:51","3:00","3:21","4:29","3:03","3:41","4:21"],
        'channel': ["ROSÉ","Lady Gaga","Reneé Rapp","Billie Eilish","Reneé Rapp","Sabrina Carpenter","Lady Gaga","Ed Sheeran","yung kai","Billie Eilish"],
        'channel_follower_count': [19200000,29600000,408000,56800000,408000,12300000,29600000,58500000,1220000,56800000],
        'publish_date': pd.date_range("2023-01-01", periods=10, freq="7D")
    }
    df = pd.DataFrame(data)
    # derived
    df['engagement_ratio'] = df['view_count'] / df['channel_follower_count']
    df['views_per_video'] = df['view_count']  # sample (per video)
    return df

# -------------------------
# Utilities
# -------------------------
def parse_duration_to_seconds(s):
    try:
        if pd.isna(s): return np.nan
        if isinstance(s,(int,float)): return float(s)
        parts = s.split(':')
        parts = [float(p) for p in parts]
        if len(parts)==2:
            return parts[0]*60 + parts[1]
        elif len(parts)==3:
            return parts[0]*3600 + parts[1]*60 + parts[2]
        else:
            return float(s)
    except:
        return np.nan

def extract_keywords(text_series, n=15):
    all_text = " ".join(text_series.fillna("").astype(str))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    c = Counter(words)
    return c.most_common(n)

def sentiment_simple(text):
    pos = ['love','happy','great','amazing','good','best','beautiful']
    neg = ['sad','angry','bad','hate','terrible','worst','mad']
    t = str(text).lower()
    p = sum(1 for w in pos if w in t)
    n = sum(1 for w in neg if w in t)
    if p>n: return 'Positive'
    if n>p: return 'Negative'
    return 'Neutral'

def safe_rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

# -------------------------
# Preprocessing (simple & robust)
# -------------------------
def preprocess_for_model(df, feature_cols, target_col, log_target=False):
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    # numeric fill
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        X[c] = X[c].fillna(X[c].median())
    # categorical -> get_dummies
    cat_cols = [c for c in X.columns if c not in numeric_cols]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    # scale numeric
    scaler = StandardScaler()
    if len(X.columns)>0:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    # target transform
    if log_target:
        y = np.log1p(y)
    return X, y, scaler

# -------------------------
# Quadrant classification (Artist comparison)
# -------------------------
def quadrant_classify(channel_df):
    # channel_df: index=channel, columns include total_views, followers, content_count
    df = channel_df.copy()
    df['views_per_video'] = df['total_views'] / df['content_count']
    df['engagement_ratio'] = df['total_views'] / df['followers']
    med_views = df['total_views'].median()
    med_eng = df['engagement_ratio'].median()
    conds = [
        (df['total_views'] > med_views) & (df['engagement_ratio'] > med_eng),
        (df['total_views'] > med_views) & (df['engagement_ratio'] <= med_eng),
        (df['total_views'] <= med_views) & (df['engagement_ratio'] > med_eng),
        (df['total_views'] <= med_views) & (df['engagement_ratio'] <= med_eng)
    ]
    choices = ['Stars','Volume Players','Engaging Niche','Developing Artists']
    df['performance_segment'] = np.select(conds, choices, default='Uncategorized')
    return df

# -------------------------
# UI - Sidebar controls
# -------------------------
st.sidebar.title("Controls")
st.sidebar.markdown("**Load data**")
uploaded = st.sidebar.file_uploader("Upload CSV file (optional)", type=['csv'])
use_sample = False
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("CSV loaded.")
    except Exception as e:
        st.sidebar.error("Failed to load CSV — using sample data. Error: " + str(e))
        df = sample_df()
        use_sample = True
else:
    df = sample_df()
    use_sample = True

# try to coerce common columns
if 'duration_string' in df.columns and 'duration' not in df.columns:
    df['duration'] = df['duration_string'].apply(parse_duration_to_seconds)
if 'publish_date' in df.columns:
    try:
        df['publish_date'] = pd.to_datetime(df['publish_date'])
    except:
        pass

# Sidebar filters
st.sidebar.markdown("### Filters")
channels = df['channel'].unique().tolist() if 'channel' in df.columns else []
sel_channels = st.sidebar.multiselect("Channels", options=channels, default=channels)
min_views = int(df['view_count'].min()) if 'view_count' in df.columns else 0
max_views = int(df['view_count'].max()) if 'view_count' in df.columns else 1
vmin, vmax = st.sidebar.slider("View count range", min_value=min_views, max_value=max_views, value=(min_views, max_views))
# Choose analysis tab
tab = st.sidebar.radio("Choose tab", ["Overview", "Artist Comparison", "Text Mining", "Predictive Modeling", "Time Series / Forecast"])

# Filter dataset based on selections
if 'channel' in df.columns and sel_channels:
    filtered = df[df['channel'].isin(sel_channels)].copy()
else:
    filtered = df.copy()
if 'view_count' in filtered.columns:
    filtered = filtered[(filtered['view_count']>=vmin) & (filtered['view_count']<=vmax)]

# -------------------------
# Overview Tab
# -------------------------
if tab=="Overview":
    st.header("Overview & EDA")
    # top metrics
    cols = st.columns(4)
    cols[0].metric("Videos", len(filtered))
    if 'view_count' in filtered.columns:
        cols[1].metric("Total Views", f"{int(filtered['view_count'].sum()):,}")
        cols[2].metric("Avg Views", f"{int(filtered['view_count'].mean()):,}")
    else:
        cols[1].metric("Total Views", "N/A")
        cols[2].metric("Avg Views", "N/A")
    cols[3].metric("Unique Channels", filtered['channel'].nunique() if 'channel' in filtered.columns else 0)

    st.subheader("Top videos")
    if 'view_count' in filtered.columns:
        top10 = filtered.nlargest(10, 'view_count')
        fig = px.bar(top10, x='title', y='view_count', color='view_count', title="Top 10 by views")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No view_count column found in data.")

    st.subheader("Duration distribution")
    if 'duration' in filtered.columns:
        fig2 = px.histogram(filtered, x='duration', nbins=20, title="Duration (seconds)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("No duration column found.")

    st.subheader("Correlation matrix (numeric)")
    num = filtered.select_dtypes(include=[np.number])
    if not num.empty:
        corr = num.corr()
        fig3 = px.imshow(corr, text_auto=True, title="Numeric correlation")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.write("No numeric columns to correlate.")

# -------------------------
# Artist Comparison Tab
# -------------------------
elif tab=="Artist Comparison":
    st.header("Artist / Channel Comparison & Quadrant Classification")
    # build channel-level stats
    if 'channel' not in df.columns:
        st.error("No 'channel' column found in dataset.")
    else:
        ch = filtered.groupby('channel').agg(
            content_count = ('title','count'),
            total_views = ('view_count','sum'),
            avg_views = ('view_count','mean'),
            followers = ('channel_follower_count','first')
        ).fillna(0)
        st.subheader("Channel summary")
        st.dataframe(ch.sort_values('total_views', ascending=False).reset_index())

        # quadrant classify
        q = quadrant_classify(ch)
        st.subheader("Quadrant classification (performance segment)")
        st.dataframe(q[['content_count','total_views','followers','views_per_video','engagement_ratio','performance_segment']].sort_values('total_views',ascending=False))

        # plot quadrant scatter
        fig = px.scatter(q.reset_index(), x='total_views', y='engagement_ratio', size='followers', color='performance_segment',
                         hover_name='channel', title="Performance Quadrant")
        st.plotly_chart(fig, use_container_width=True)

        # show top by segment
        st.subheader("Top channels per segment")
        seg = q.reset_index().groupby('performance_segment').apply(lambda d: d.sort_values('total_views', ascending=False).head(3))
        st.dataframe(seg.reset_index(drop=True)[['channel','total_views','followers','performance_segment']])

# -------------------------
# Text Mining Tab
# -------------------------
elif tab=="Text Mining":
    st.header("Text Mining & Simple NLP")
    text_source_opt = st.selectbox("Source column for text analysis", options=[c for c in df.columns if df[c].dtype==object], index=0) if any(df.dtypes==object) else None
    if text_source_opt is None:
        st.error("No text column found in dataset.")
    else:
        st.subheader(f"Top keywords from '{text_source_opt}'")
        kw = extract_keywords(filtered[text_source_opt], n=20)
        kw_df = pd.DataFrame(kw, columns=['keyword','freq'])
        fig = px.bar(kw_df, x='freq', y='keyword', orientation='h', title="Top keywords")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sentiment (very basic wordmatch) on descriptions")
        if 'description' in filtered.columns:
            filtered['sentiment'] = filtered['description'].apply(sentiment_simple)
            s_counts = filtered['sentiment'].value_counts()
            fig2 = px.pie(values=s_counts.values, names=s_counts.index, title="Sentiment distribution")
            st.plotly_chart(fig2, use_container_width=True)
            st.subheader("Sentiment vs Views")
            if 'view_count' in filtered.columns:
                fig3 = px.box(filtered, x='sentiment', y='view_count', title="Views by Sentiment")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.write("No 'description' column to analyze sentiment.")

# -------------------------
# Predictive Modeling Tab
# -------------------------
elif tab=="Predictive Modeling":
    st.header("Predictive Modeling — compare algorithms")
    st.markdown("Select features and target, choose algorithms to train and compare. Target can be raw views or log-transformed.")

    # feature selection UI
    all_cols = filtered.columns.tolist()
    default_features = [c for c in ['channel_follower_count','duration','engagement_ratio','release_month','views_per_video'] if c in all_cols]
    feature_cols = st.multiselect("Feature columns (X)", options=[c for c in all_cols if c!='view_count'], default=default_features)
    if 'view_count' not in filtered.columns:
        st.error("No 'view_count' target column found. Cannot model.")
    else:
        target_col = 'view_count'
        log_target = st.checkbox("Log-transform target (recommended)", value=True)
        test_size = st.slider("Test size (%)", 10, 50, 30)/100.0
        random_state = 42

        if len(feature_cols)==0:
            st.warning("Select at least one feature.")
        else:
            X, y, scaler = preprocess_for_model(filtered, feature_cols, target_col, log_target=log_target)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            st.write("Training samples:", len(X_train), "Test samples:", len(X_test))

            # choose algorithms
            algos = {
                "Linear": LinearRegression(),
                "Ridge": Ridge(random_state=random_state),
                "Lasso": Lasso(random_state=random_state),
                "ElasticNet": ElasticNet(random_state=random_state),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR(),
                "RandomForest": RandomForestRegressor(n_estimators=200, random_state=random_state),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=random_state)
            }
            if HAS_XGB:
                algos["XGBoost"] = xgb.XGBRegressor(n_estimators=200, random_state=random_state, verbosity=0)
            if HAS_LGB:
                algos["LightGBM"] = lgb.LGBMRegressor(n_estimators=200, random_state=random_state)

            chosen = st.multiselect("Choose algorithms to run", options=list(algos.keys()), default=list(algos.keys()))
            if st.button("Train & Evaluate"):
                results = []
                kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
                for name in chosen:
                    model = algos[name]
                    # cross-validated R2 (on transformed target if log_target)
                    try:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
                        cv_mean = np.mean(cv_scores)
                    except Exception:
                        cv_mean = np.nan
                    # fit
                    try:
                        model.fit(X_train, y_train)
                        ypred = model.predict(X_test)
                        # invert transform if necessary for metrics
                        if log_target:
                            y_test_orig = np.expm1(y_test)
                            ypred_orig = np.expm1(ypred)
                        else:
                            y_test_orig = y_test
                            ypred_orig = ypred
                        mae = mean_absolute_error(y_test_orig, ypred_orig)
                        rmse = safe_rmse(y_test_orig, ypred_orig)
                        r2 = r2_score(y_test, ypred)  # r2 on transformed target
                        results.append({'model':name, 'cv_r2':cv_mean, 'test_r2_transformed':r2, 'test_mae':mae, 'test_rmse':rmse, 'model_obj':model})
                    except Exception as e:
                        results.append({'model':name, 'cv_r2':cv_mean, 'error':str(e)})

                res_df = pd.DataFrame(results).sort_values('test_r2_transformed', ascending=False).reset_index(drop=True)
                st.subheader("Model comparison")
                display_df = res_df[['model','cv_r2','test_r2_transformed','test_mae','test_rmse']].copy()
                st.dataframe(display_df.fillna("N/A"))

                best = results[0] if results else None
                if not res_df.empty:
                    best_name = res_df.loc[0,'model']
                    st.success(f"Best model by R² (transformed target): {best_name}")
                    # show feature importance if available
                    best_obj = res_df.loc[0,'model_obj']
                    if hasattr(best_obj, 'feature_importances_'):
                        fi = pd.Series(best_obj.feature_importances_, index=X.columns).sort_values(ascending=False)
                        st.subheader("Feature importances")
                        fig = px.bar(fi.reset_index(), x=0, y='index', orientation='h', labels={'index':'feature',0:'importance'})
                        st.plotly_chart(fig, use_container_width=True)
                    elif hasattr(best_obj, 'coef_'):
                        coefs = pd.Series(best_obj.coef_, index=X.columns).sort_values(ascending=False)
                        st.subheader("Model coefficients")
                        st.dataframe(coefs)
                    # show predictions vs actual
                    st.subheader("Predictions (test set) for best model")
                    preds = best_obj.predict(X_test)
                    if log_target:
                        preds_orig = np.expm1(preds)
                        ytest_orig = np.expm1(y_test)
                    else:
                        preds_orig = preds
                        ytest_orig = y_test
                    outdf = pd.DataFrame({'actual': ytest_orig, 'predicted': preds_orig})
                    outdf = outdf.reset_index(drop=True)
                    figp = px.scatter(outdf, x='actual', y='predicted', trendline='ols', title='Actual vs Predicted')
                    st.plotly_chart(figp, use_container_width=True)

# -------------------------
# Time Series / Forecast Tab
# -------------------------
elif tab=="Time Series / Forecast":
    st.header("Time series aggregation & forecasting")
    # need publish_date and view_count
    if 'publish_date' not in df.columns or 'view_count' not in df.columns:
        st.error("Data needs 'publish_date' (datetime) and 'view_count' columns for time series forecasting.")
    else:
        ts = filtered[['publish_date','view_count']].copy()
        ts = ts.dropna()
        ts = ts.groupby('publish_date').sum().reset_index().sort_values('publish_date')
        st.subheader("Aggregated time series (daily)")
        st.dataframe(ts.head(10))
        # resample daily
        ts_daily = ts.set_index('publish_date').resample('D').sum().fillna(0).reset_index()
        st.line_chart(ts_daily.rename(columns={'publish_date':'index'}).set_index('index')['view_count'])

        # forecast
        if HAS_PROPHET:
            st.subheader("Forecast with Prophet")
            dfp = ts_daily.rename(columns={'publish_date':'ds','view_count':'y'})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            with st.spinner("Fitting Prophet..."):
                model.fit(dfp)
                periods = st.slider("Forecast days", 7, 365, 90)
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)
                fig = model.plot(forecast)
                st.pyplot(fig)
                # show forecast tail
                st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail())
        else:
            st.info("Prophet not installed — falling back to 7-day moving average forecast")
            ts_daily['ma7'] = ts_daily['view_count'].rolling(7,min_periods=1).mean()
            st.line_chart(ts_daily.set_index('publish_date')[['view_count','ma7']])
            last_ma = ts_daily['ma7'].iloc[-1]
            periods = st.slider("Forecast days (simple MA)", 7, 180, 30)
            forecast_idx = pd.date_range(ts_daily['publish_date'].max()+pd.Timedelta(days=1), periods=periods)
            forecast_vals = [last_ma]*periods
            fdf = pd.DataFrame({'ds':forecast_idx, 'yhat':forecast_vals})
            st.dataframe(fdf.head())

# -------------------------
# Footer + download
# -------------------------
st.sidebar.markdown("---")
st.sidebar.write("⚙️ Made by mostafa mohamed ")

st.markdown("---")
st.markdown("### Download filtered data")
csv = filtered.to_csv(index=False)
st.download_button("Download filtered CSV", csv, file_name="filtered_youtube.csv", mime="text/csv")

st.markdown("App finished. If you want I can further: (1) add more algorithms (XGBoost tuning, LightGBM), (2) add stacking/ensembling, (3) store models, or (4) run on your CSV and return result files — tell me which.")
