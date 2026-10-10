import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from quant_research.data.caching import load_from_cache
from quant_research.backtesting.engine import BacktestEngine
from quant_research.portfolio.policies import simple_threshold_policy, mdp_heuristic_policy
from quant_research.data.news_data import KaggleNewsProvider
from quant_research.config.settings import load_settings

st.set_page_config(page_title="Quant Research System", layout="wide", initial_sidebar_state="expanded")

# --- DATA LOADING ---
@st.cache_data
def load_report():
    path = Path("reports/experiment_summaries/final_report.json")
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_all_predictions():
    preds = {}
    for exp in ['A', 'B', 'C', 'D']:
        df = load_from_cache(f"experiment_{exp}_predictions", stage="processed")
        if df is not None:
            preds[exp] = df
    return preds

@st.cache_data
def get_backtest_curves(features_df, all_preds):
    engine = BacktestEngine(initial_capital=100000, tc_bps=5.0)
    curves = {}
    for exp, preds in all_preds.items():
        if exp == 'D':
            policy = mdp_heuristic_policy(preds, features_df)
        else:
            policy = simple_threshold_policy(preds)
        bt_results = engine.run(features_df, preds, policy)
        curves[exp] = bt_results
    return curves

report = load_report()
features_df = load_from_cache("features_pipeline_output", stage="features")
all_preds = load_all_predictions()
if features_df is not None and all_preds:
    curves = get_backtest_curves(features_df, all_preds)
else:
    curves = {}

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("System Navigation")
page = st.sidebar.radio("Select View:", [
    "1. Overview & Strategy Definitions", 
    "2. Comparative Analysis", 
    "3. Strategy Deep Dive",
    "4. NLP & News Impact",
    "5. Regime Detection (HMM)"
])

st.sidebar.markdown("---")
st.sidebar.info("This system investigates whether financial news sentiment, earnings data, and HMM regime detection improve predictive accuracy and portfolio performance.")

# --- PAGE 1: OVERVIEW ---
if page == "1. Overview & Strategy Definitions":
    st.title("System Overview & Strategy Definitions")
    st.markdown("""
    ### The Research Question
    *Do financial-news sentiment and company-specific financial results improve out-of-sample stock-direction prediction across different hidden market regimes, and do any predictive improvements translate into superior risk-adjusted portfolio performance after realistic trading costs?*
    
    ### Strategy Definitions
    - **Strategy A (Price-Only Baseline)**: Uses purely autoregressive market data. Features include lagged returns (1d, 5d, 21d), rolling volatility (21d), moving average crossovers (21d vs 63d), and volume metrics.
    - **Strategy B (Price + News Sentiment)**: Adds aggregated daily FinBERT sentiment scores derived from financial news articles. Features include news volume, average daily sentiment, and sentiment dispersion.
    - **Strategy C (Price + Earnings)**: Adds fundamental momentum via quarterly earnings surprises. Features include EPS actuals, EPS consensus estimates, and Revenue. Information is strictly forward-filled without lookahead bias.
    - **Strategy D (Combined + HMM MDP Policy)**: The master strategy. It combines Price, News, and Earnings features. Crucially, it employs a **Hidden Markov Model (HMM)** to detect latent market regimes (e.g., high vs low volatility). Instead of a naive threshold, it uses a **Markov Decision Process (MDP) heuristic** to dynamically size down exposure during high-risk regimes.
    """)
    
    if report:
        st.subheader("Summary Metrics")
        acc_data = {f"Strategy {k}": v['accuracy'] for k, v in report['predictive_metrics'].items()}
        sharpe_data = {f"Strategy {k}": v['sharpe_ratio'] for k, v in report['portfolio_performance'].items()}
        
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(x=list(acc_data.keys()), y=list(acc_data.values()), labels={'x':'Strategy', 'y':'Accuracy'}, title="Out-of-Sample Accuracy", color=list(acc_data.values()), color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(x=list(sharpe_data.keys()), y=list(sharpe_data.values()), labels={'x':'Strategy', 'y':'Sharpe Ratio'}, title="Risk-Adjusted Portfolio Return (Sharpe)", color=list(sharpe_data.values()), color_continuous_scale="Greens")
            st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: COMPARATIVE ANALYSIS ---
elif page == "2. Comparative Analysis":
    st.title("Comparative Portfolio Analysis")
    if curves:
        st.markdown("### Simulated Equity Curves (Out-of-Sample)")
        
        # Combine equity curves
        combined_equity = pd.DataFrame({f"Strategy {exp}": df['equity'] for exp, df in curves.items()})
        fig_eq = px.line(combined_equity, title="Portfolio Value Over Time ($100,000 Initial Capital)", labels={'value': 'Portfolio Value ($)', 'Date': 'Date'})
        st.plotly_chart(fig_eq, use_container_width=True, height=600)
        
        st.markdown("### Drawdown Profiles")
        combined_dd = pd.DataFrame({f"Strategy {exp}": df['drawdown'] for exp, df in curves.items()})
        fig_dd = px.line(combined_dd, title="Drawdown Over Time", labels={'value': 'Drawdown %', 'Date': 'Date'})
        fig_dd.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_dd, use_container_width=True)
        
        st.markdown("### Detailed Metrics Table")
        df_perf = pd.DataFrame(report['portfolio_performance']).T
        df_perf.index = [f"Strategy {x}" for x in df_perf.index]
        st.dataframe(df_perf.style.highlight_max(color='lightgreen', subset=['annualized_return', 'sharpe_ratio']).highlight_min(color='lightgreen', subset=['max_drawdown']))

# --- PAGE 3: STRATEGY DEEP DIVE ---
elif page == "3. Strategy Deep Dive":
    st.title("Strategy Deep Dive")
    selected_exp = st.selectbox("Select Strategy", ["A", "B", "C", "D"])
    
    if all_preds and selected_exp in all_preds:
        preds = all_preds[selected_exp]
        bt_results = curves[selected_exp]
        
        st.subheader(f"Strategy {selected_exp} Execution Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Test Predictions", len(preds))
        with col2:
            st.metric("Accuracy", f"{report['predictive_metrics'][selected_exp]['accuracy']:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{report['portfolio_performance'][selected_exp]['sharpe_ratio']:.2f}")
            
        st.markdown("### Prediction Confidence Distribution")
        fig_hist = px.histogram(preds, x="prob", color="target", barmode="overlay", title="Predicted Probability vs Actual Target", labels={'prob': 'Predicted Probability of Positive Return', 'target': 'Actual Label (1=Positive)'}, opacity=0.7)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("### Daily Returns & Turnover")
        fig_rets = go.Figure()
        fig_rets.add_trace(go.Bar(x=bt_results.index, y=bt_results['net_return'], name="Net Daily Return"))
        fig_rets.update_layout(title="Daily Net Returns")
        st.plotly_chart(fig_rets, use_container_width=True)

# --- PAGE 4: NLP & NEWS IMPACT ---
elif page == "4. NLP & News Impact":
    st.title("NLP Engine & News Impact")
    st.markdown("""
    ### FinBERT Processing Pipeline
    The system utilizes `ProsusAI/finbert`, a BERT model pre-trained on financial corpora, to classify incoming news headlines into `Positive`, `Negative`, or `Neutral` states.
    
    1. **Ingestion**: News is ingested with a strict timestamp.
    2. **Inference**: The model returns softmax probabilities: $P(pos), P(neg), P(neu)$.
    3. **Scoring**: A signed sentiment score is calculated as $P(pos) - P(neg)$.
    4. **Aggregation**: Articles published after market close are strictly assigned to the **next trading day** to prevent lookahead bias.
    """)
    
    st.markdown("### Example: Actual Historical News Flow")
    st.info("The system now fetches REAL historical Indian financial news via the Kaggle dataset. Below is a sample of actual news events that fed into the FinBERT pipeline.")
    settings = load_settings()
    n_provider = KaggleNewsProvider(["^NSEI"], "2020-01-01", "2020-01-31") # Just fetch 1 month for UI speed
    sample_news = n_provider.fetch_news()
    
    if not sample_news.empty:
        # Sort and take top 20 to avoid heavy UI latency
        sample_news = sample_news.sort_values('publication_timestamp', ascending=False).head(20)
        
        if 'FinBERT_Score' not in sample_news.columns:
            from quant_research.nlp.finbert import FinBERTSentimentAnalyzer
            from quant_research.config.settings import load_settings
            import pandas as pd
            import streamlit as st
            
            with st.spinner("Calculating live FinBERT scores for the sample..."):
                settings = load_settings()
                finbert = FinBERTSentimentAnalyzer(settings.nlp)
                # Run inference on the 20 sample headlines
                results = finbert.analyze(sample_news['headline'].tolist())
                # Extract sentiment_score
                sample_news['FinBERT_Score'] = [r['sentiment_score'] for r in results]
            
        st.dataframe(sample_news[['publication_timestamp', 'headline', 'FinBERT_Score', 'article_text']])
        
        st.markdown("### Impact of Extreme News")
        st.write("Real news articles with high absolute FinBERT scores create spikes in the daily `sentiment_dispersion` and `avg_sentiment` features, prompting Strategies B and D to adjust their predictive probabilities.")
        
        fig_scatter = px.scatter(sample_news, x="publication_timestamp", y="FinBERT_Score", color="FinBERT_Score", color_continuous_scale="RdYlGn", title="Sentiment Timeline")
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- PAGE 5: REGIME DETECTION (HMM) ---
elif page == "5. Regime Detection (HMM)":
    st.title("Latent Market Regimes via HMM")
    st.markdown("""
    ### Hidden Markov Model (HMM)
    Financial markets exhibit structural breaks and shifting regimes (e.g., bull vs. bear markets, low vs. high volatility). Strategy D utilizes a Gaussian HMM to detect these latent states using historical returns and realized volatility.
    
    *Crucially, the HMM is fit entirely within the walk-forward training window. Out-of-sample periods are labeled using the filtered probabilities (predicting the current state using strictly past data).*
    """)
    
    if 'D' in all_preds and 'hmm_high_vol_prob' in all_preds['D'].columns:
        st.markdown("### HMM High Volatility State Probability vs Market Price")
        
        # Merge price and HMM prob
        df_D = all_preds['D'].reset_index()
        df_feat = features_df.reset_index()
        plot_df = pd.merge(df_D[['Date', 'hmm_high_vol_prob']], df_feat[['Date', 'Close']], on='Date', how='inner')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Close'], name="AAPL Close Price", mode='lines', yaxis="y1"))
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['hmm_high_vol_prob'], name="P(High Volatility Regime)", mode='lines', fill='tozeroy', opacity=0.3, yaxis="y2", marker=dict(color="red")))
        
        fig.update_layout(
            title="Market Regimes Over Time",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Probability", overlaying="y", side="right", range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Mechanism in the MDP Policy")
        st.info("The Markov Decision Process (MDP) heuristic observes `P(High Volatility Regime)`. As this probability approaches 1.0, the MDP aggressively scales down the portfolio exposure, reducing drawdowns during structural market crashes. This directly drives the superior Sharpe Ratio seen in Strategy D.")
    else:
        st.warning("HMM probabilities not found in the current feature set.")
