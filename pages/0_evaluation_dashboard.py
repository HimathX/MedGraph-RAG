"""
RAGAS Evaluation Dashboard

A separate Streamlit page for viewing and analyzing evaluation metrics over time.
This dashboard provides insights into system performance and helps identify patterns
in answer quality, faithfulness, and relevancy.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.evaluation import EvaluationLogger
from datetime import datetime, timedelta

# Page Config
st.set_page_config(
    page_title="RAGAS Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    h1 {
        text-transform: uppercase;
        font-weight: 900 !important;
        letter-spacing: -2px;
    }
    .metric-card {
        border: 2px solid #000;
        border-radius: 0px;
        box-shadow: 4px 4px 0px #000;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 RAGAS Evaluation Dashboard")
st.caption("Monitor and analyze RAG system performance over time")

# Initialize logger
logger = EvaluationLogger()

# Load evaluation history
df = logger.get_evaluation_history()

if df.empty:
    st.info("No evaluation data available yet. Enable RAGAS evaluation in the main app to start collecting metrics.")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    # Date range filter
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['timestamp'].dt.date >= start_date) & 
                   (df['timestamp'].dt.date <= end_date)]
    
    # Metric threshold filter
    st.subheader("Metric Filters")
    min_faithfulness = st.slider("Min Faithfulness", 0.0, 1.0, 0.0, 0.05)
    min_relevancy = st.slider("Min Answer Relevancy", 0.0, 1.0, 0.0, 0.05)
    
    # Apply filters
    if 'faithfulness' in df.columns:
        df = df[df['faithfulness'] >= min_faithfulness]
    if 'answer_relevancy' in df.columns:
        df = df[df['answer_relevancy'] >= min_relevancy]
    
    st.divider()
    st.caption(f"Showing {len(df)} evaluations")

# Main Dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Metrics Analysis", "🔍 Query Explorer", "📋 Raw Data"])

with tab1:
    st.header("Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Evaluations",
            len(df),
            help="Total number of evaluated queries"
        )
    
    with col2:
        if 'faithfulness' in df.columns:
            avg_faithfulness = df['faithfulness'].mean()
            st.metric(
                "Avg Faithfulness",
                f"{avg_faithfulness:.3f}",
                help="Average faithfulness score across all queries"
            )
    
    with col3:
        if 'answer_relevancy' in df.columns:
            avg_relevancy = df['answer_relevancy'].mean()
            st.metric(
                "Avg Relevancy",
                f"{avg_relevancy:.3f}",
                help="Average answer relevancy score"
            )
    
    with col4:
        if 'faithfulness' in df.columns:
            hallucination_rate = (df['faithfulness'] < 0.7).sum() / len(df) * 100
            st.metric(
                "Hallucination Rate",
                f"{hallucination_rate:.1f}%",
                help="Percentage of answers with faithfulness < 0.7"
            )
    
    st.divider()
    
    # Metrics over time
    if 'timestamp' in df.columns and 'faithfulness' in df.columns:
        st.subheader("Metrics Over Time")
        
        # Prepare data for plotting
        df_sorted = df.sort_values('timestamp')
        
        fig = go.Figure()
        
        if 'faithfulness' in df.columns:
            fig.add_trace(go.Scatter(
                x=df_sorted['timestamp'],
                y=df_sorted['faithfulness'],
                mode='lines+markers',
                name='Faithfulness',
                line=dict(color='#4ECDC4', width=2),
                marker=dict(size=6)
            ))
        
        if 'answer_relevancy' in df.columns:
            fig.add_trace(go.Scatter(
                x=df_sorted['timestamp'],
                y=df_sorted['answer_relevancy'],
                mode='lines+markers',
                name='Answer Relevancy',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=6)
            ))
        
        # Add threshold line
        fig.add_hline(y=0.7, line_dash="dash", line_color="gray",
                     annotation_text="Threshold (0.7)")
        
        fig.update_layout(
            title="Evaluation Metrics Timeline",
            xaxis_title="Time",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Metrics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Faithfulness distribution
        if 'faithfulness' in df.columns:
            st.subheader("Faithfulness Distribution")
            
            fig = px.histogram(
                df,
                x='faithfulness',
                nbins=20,
                color_discrete_sequence=['#4ECDC4']
            )
            fig.update_layout(
                xaxis_title="Faithfulness Score",
                yaxis_title="Count",
                showlegend=False,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.caption(f"Mean: {df['faithfulness'].mean():.3f} | "
                      f"Median: {df['faithfulness'].median():.3f} | "
                      f"Std: {df['faithfulness'].std():.3f}")
    
    with col2:
        # Answer relevancy distribution
        if 'answer_relevancy' in df.columns:
            st.subheader("Answer Relevancy Distribution")
            
            fig = px.histogram(
                df,
                x='answer_relevancy',
                nbins=20,
                color_discrete_sequence=['#FF6B6B']
            )
            fig.update_layout(
                xaxis_title="Answer Relevancy Score",
                yaxis_title="Count",
                showlegend=False,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.caption(f"Mean: {df['answer_relevancy'].mean():.3f} | "
                      f"Median: {df['answer_relevancy'].median():.3f} | "
                      f"Std: {df['answer_relevancy'].std():.3f}")
    
    # Correlation analysis
    if 'faithfulness' in df.columns and 'answer_relevancy' in df.columns:
        st.divider()
        st.subheader("Metric Correlation")
        
        fig = px.scatter(
            df,
            x='faithfulness',
            y='answer_relevancy',
            color='num_contexts' if 'num_contexts' in df.columns else None,
            size='num_contexts' if 'num_contexts' in df.columns else None,
            hover_data=['question'] if 'question' in df.columns else None,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title="Faithfulness",
            yaxis_title="Answer Relevancy",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        correlation = df['faithfulness'].corr(df['answer_relevancy'])
        st.caption(f"Correlation coefficient: {correlation:.3f}")

with tab3:
    st.header("Query Explorer")
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        ["timestamp", "faithfulness", "answer_relevancy"],
        index=0
    )
    sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
    
    df_sorted = df.sort_values(
        sort_by,
        ascending=(sort_order == "Ascending")
    )
    
    # Display queries
    for idx, row in df_sorted.iterrows():
        with st.expander(
            f"Query: {row.get('question', 'N/A')[:100]}... "
            f"(F: {row.get('faithfulness', 0):.2f}, R: {row.get('answer_relevancy', 0):.2f})"
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Question:**")
                st.write(row.get('question', 'N/A'))
                
                st.markdown("**Answer:**")
                st.write(row.get('answer', 'N/A'))
            
            with col2:
                st.markdown("**Metrics:**")
                if 'faithfulness' in row:
                    st.metric("Faithfulness", f"{row['faithfulness']:.3f}")
                if 'answer_relevancy' in row:
                    st.metric("Answer Relevancy", f"{row['answer_relevancy']:.3f}")
                if 'num_contexts' in row:
                    st.metric("Contexts Used", int(row['num_contexts']))
                if 'timestamp' in row:
                    st.caption(f"Time: {row['timestamp']}")

with tab4:
    st.header("Raw Data")
    
    # Display dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"ragas_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Summary statistics
    st.divider()
    st.subheader("Summary Statistics")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        st.dataframe(
            df[numeric_cols].describe(),
            use_container_width=True
        )
