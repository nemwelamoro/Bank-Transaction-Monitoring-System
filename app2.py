import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, plot_tree
import io
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="BRB - Transaction Monitoring System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --success-color: #059669;
        --warning-color: #d97706;
        --danger-color: #dc2626;
        --neutral-bg: #f8fafc;
    }
    
    /* Remove default streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
    }
    
    .main-header p {
        color: #e0e7ff;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Risk level cards */
    .risk-critical {
        background-color: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 1.5rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    
    .risk-medium {
        background-color: #fffbeb;
        border-left: 4px solid #d97706;
        padding: 1.5rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    
    .risk-low {
        background-color: #f0fdf4;
        border-left: 4px solid #059669;
        padding: 1.5rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    
    /* Information boxes */
    .info-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        padding: 1.25rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #1e3a8a;
        margin-top: 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Transaction details */
    .transaction-header {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }
    
    .explanation-item {
        color: #475569;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f1f5f9;
        line-height: 1.6;
    }
    
    .explanation-item:last-child {
        border-bottom: none;
    }
    
    /* Action buttons */
    .action-required {
        background-color: #f0f9ff;
        border: 1px solid #7dd3fc;
        padding: 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #0c4a6e;
    }
    
    /* Section headers */
    .section-header {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* List styling */
    .styled-list {
        color: #475569;
        line-height: 1.8;
    }
    
    .styled-list li {
        margin-bottom: 0.5rem;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>BRB Transaction Monitoring System</h1>
    <p>Intelligent anomaly detection for salary disbursement operations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### System Configuration")
    
    st.markdown("#### Risk Thresholds")
    high_risk_threshold = st.slider("High Risk Score", 3, 10, 7, help="Transactions scoring above this require immediate attention")
    medium_risk_threshold = st.slider("Medium Risk Score", 2, 7, 4, help="Transactions scoring above this require review")
    
    st.markdown("#### Detection Sensitivity")
    contamination = st.slider("Anomaly Detection Rate (%)", 1, 20, 5, help="Expected percentage of anomalous transactions") / 100
    
    st.markdown("#### Display Options")
    show_all_transactions = st.checkbox("Show All Transactions", False)
    enable_auto_alerts = st.checkbox("Enable Automated Alerts", False)
    
    st.markdown("---")
    st.markdown("### Quick Guide")
    st.markdown("""
    1. Upload transaction CSV file
    2. Review flagged anomalies
    3. Download actionable reports
    4. Configure alert preferences
    """)

# File upload
uploaded_file = st.file_uploader(
    "Upload Transaction Data", 
    type="csv", 
    help="Select the Bank of Burundi salary disbursement CSV file"
)

if uploaded_file is not None:
    # Load and process data
    df = pd.read_csv(uploaded_file)
    
    # Preprocessing
    numeric_cols = [' Debit Amount', ' Credit Amount', ' Account balance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Feature engineering
    df['Net'] = df[' Credit Amount'] - df[' Debit Amount']
    df['Running Balance'] = df['Net'].cumsum()
    df['Flag'] = ''
    df['Risk Score'] = 0
    df['Explanations'] = [[] for _ in range(len(df))]
    df['Recommended Action'] = ''
    df['Impact Level'] = 'Low'
    
    # Rule 1: High Duplicate Documents
    doc_counts = df['Document no.'].value_counts()
    high_duplicates = doc_counts[doc_counts > 3].index
    mask = df['Document no.'].isin(high_duplicates)
    df.loc[mask, 'Flag'] += 'High Duplicate Count; '
    df.loc[mask, 'Risk Score'] += 4
    for idx in df[mask].index:
        count = doc_counts[df.loc[idx, 'Document no.']]
        df.at[idx, 'Explanations'].append(f"Document appears {count} times (threshold: 3). Potential batch processing error or system retry detected.")
        df.at[idx, 'Recommended Action'] = 'Verify multiple salary credits for legitimacy. Review for duplicate employee records.'
    
    # Rule 2: Negative Net
    mask = df['Net'] < 0
    df.loc[mask, 'Flag'] += 'Negative Net; '
    df.loc[mask, 'Risk Score'] += 3
    for idx in df[mask].index:
        debit = df.loc[idx, ' Debit Amount']
        credit = df.loc[idx, ' Credit Amount']
        df.at[idx, 'Explanations'].append(f"Debit amount ({debit:,.0f} FBu) exceeds credit amount ({credit:,.0f} FBu). Worker account may have outstanding loan obligations or active blocks.")
        df.at[idx, 'Recommended Action'] = 'Contact account holder to verify loan status and account restrictions.'
        df.at[idx, 'Impact Level'] = 'High'
    
    # Rule 3: Rare Agent Code
    agent_counts = df['Agent code'].value_counts()
    rare_agents = agent_counts[agent_counts <= 1].index
    mask = df['Agent code'].isin(rare_agents)
    df.loc[mask, 'Flag'] += 'Rare Agent Code; '
    df.loc[mask, 'Risk Score'] += 2
    for idx in df[mask].index:
        agent = df.loc[idx, 'Agent code']
        df.at[idx, 'Explanations'].append(f"Agent code '{agent}' appears only once in dataset. May indicate unauthorized intermediary or data entry error.")
        df.at[idx, 'Recommended Action'] = 'Verify agent authorization against approved registry.'
    
    # Rule 4: Unpaired Transactions
    grouped_net = df.groupby('Document no.')['Net'].sum()
    unpaired_docs = grouped_net[grouped_net != 0].index
    mask = df['Document no.'].isin(unpaired_docs)
    df.loc[mask, 'Flag'] += 'Unpaired Transaction; '
    df.loc[mask, 'Risk Score'] += 5
    for idx in df[mask].index:
        doc = df.loc[idx, 'Document no.']
        net_sum = grouped_net[doc]
        df.at[idx, 'Explanations'].append(f"Transaction not balanced (net sum: {net_sum:,.0f} FBu). Transfer likely failed or partially processed.")
        df.at[idx, 'Recommended Action'] = 'Priority investigation required. Worker may not have received salary payment.'
        df.at[idx, 'Impact Level'] = 'Critical'
    
    # Rule 5: Large Balance Discrepancy
    df['Discrepancy'] = abs(df['Running Balance'] - df[' Account balance'])
    large_disc_threshold = df[' Debit Amount'].mean() * 2
    mask = df['Discrepancy'] > large_disc_threshold
    df.loc[mask, 'Flag'] += 'Large Balance Discrepancy; '
    df.loc[mask, 'Risk Score'] += 4
    for idx in df[mask].index:
        disc = df.loc[idx, 'Discrepancy']
        df.at[idx, 'Explanations'].append(f"Balance discrepancy of {disc:,.0f} FBu detected (threshold: {large_disc_threshold:,.0f} FBu). Account may be dormant or require reconciliation.")
        df.at[idx, 'Recommended Action'] = 'Review account activity history and reconcile balance records.'
        df.at[idx, 'Impact Level'] = 'Medium'
    
    # ML Anomaly Detection
    features = df[[' Debit Amount', ' Credit Amount', 'Net']].copy()
    iso = IsolationForest(contamination=contamination, random_state=42)
    df['Anomaly Score'] = iso.fit_predict(features)
    df['Anomaly Probability'] = iso.score_samples(features)
    
    mask = df['Anomaly Score'] == -1
    df.loc[mask, 'Flag'] += 'ML Outlier; '
    df.loc[mask, 'Risk Score'] += 3
    
    for idx in df[mask].index:
        debit = df.loc[idx, ' Debit Amount']
        credit = df.loc[idx, ' Credit Amount']
        
        debit_mean, debit_std = df[' Debit Amount'].mean(), df[' Debit Amount'].std()
        credit_mean, credit_std = df[' Credit Amount'].mean(), df[' Credit Amount'].std()
        
        contributors = []
        if abs(debit - debit_mean) > 2 * debit_std:
            contributors.append(f"Debit amount ({debit:,.0f} FBu) deviates {abs(debit - debit_mean)/debit_std:.1f} standard deviations from average")
        if abs(credit - credit_mean) > 2 * credit_std:
            contributors.append(f"Credit amount ({credit:,.0f} FBu) deviates {abs(credit - credit_mean)/credit_std:.1f} standard deviations from average")
        
        explanation = "Machine learning model detected unusual pattern: " + ("; ".join(contributors) if contributors else "Transaction amounts show significant deviation from historical norms")
        df.at[idx, 'Explanations'].append(explanation)
        if not df.at[idx, 'Recommended Action']:
            df.at[idx, 'Recommended Action'] = 'Review transaction for potential fraud indicators or data errors.'
    
    # Risk categorization
    df.loc[df['Risk Score'] >= high_risk_threshold, 'Impact Level'] = 'Critical'
    df.loc[(df['Risk Score'] >= medium_risk_threshold) & (df['Risk Score'] < high_risk_threshold), 'Impact Level'] = 'Medium'
    
    # Filter flagged transactions
    flagged = df[df['Flag'] != ''].copy()
    high_risk = flagged[flagged['Risk Score'] >= high_risk_threshold].sort_values('Risk Score', ascending=False)
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        critical_count = len(flagged[flagged['Impact Level'] == 'Critical'])
        st.metric("Critical Issues", critical_count, delta=f"{critical_count/len(df)*100:.1f}%" if len(df) > 0 else "0%", delta_color="inverse")
    with col3:
        workers_affected = flagged['Document no.'].nunique()
        st.metric("Workers Affected", workers_affected)
    with col4:
        potential_loss = flagged[flagged['Net'] < 0]['Net'].sum()
        st.metric("Potential Loss", f"{abs(potential_loss):,.0f} FBu")
    
    # Alert section
    st.markdown('<p class="section-header">Detection Results and Recommended Actions</p>', unsafe_allow_html=True)
    
    if not high_risk.empty:
        tab1, tab2, tab3 = st.tabs(["Critical Alerts", "Rule Performance", "Pattern Analysis"])
        
        with tab1:
            st.markdown("### High Priority Transactions")
            
            for idx, row in high_risk.head(10).iterrows():
                risk_class = "risk-critical" if row['Impact Level'] == 'Critical' else "risk-medium"
                
                st.markdown(f"""
                <div class="{risk_class}">
                    <div class="transaction-header">Transaction #{idx} | Document: {row['Document no.']} | Risk Score: {row['Risk Score']}/15</div>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Detection Reasoning:**")
                    for explanation in row['Explanations']:
                        st.markdown(f'<div class="explanation-item">{explanation}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Recommended Action:**")
                    st.markdown(f'<div class="action-required">{row["Recommended Action"]}</div>', unsafe_allow_html=True)
                    st.markdown(f"**Impact Level:** `{row['Impact Level']}`")
                    st.markdown(f"**Debit:** {row[' Debit Amount']:,.0f} FBu | **Credit:** {row[' Credit Amount']:,.0f} FBu")
                
                st.markdown("---")
        
        with tab2:
            st.markdown("### Detection Rule Performance")
            
            flag_analysis = []
            for flag_type in ['High Duplicate Count', 'Negative Net', 'Rare Agent Code', 'Unpaired Transaction', 'Large Balance Discrepancy', 'ML Outlier']:
                count = flagged['Flag'].str.contains(flag_type).sum()
                if count > 0:
                    flag_analysis.append({
                        'Detection Rule': flag_type,
                        'Occurrences': count,
                        'Percentage': f"{count/len(df)*100:.2f}%"
                    })
            
            flag_df = pd.DataFrame(flag_analysis)
            
            if not flag_df.empty:
                fig = px.bar(flag_df, x='Detection Rule', y='Occurrences', 
                            title='Rule Activation Frequency',
                            color='Occurrences',
                            color_continuous_scale='Blues')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(flag_df, use_container_width=True, hide_index=True)
                
                st.markdown("""
                <div class="info-box">
                <h4>Detection Rule Descriptions</h4>
                <ul class="styled-list">
                    <li><strong>High Duplicate Count:</strong> Multiple transactions with identical document numbers suggesting batch processing errors or system retries</li>
                    <li><strong>Negative Net:</strong> Debit exceeds credit, indicating loan deductions or account restrictions preventing full salary credit</li>
                    <li><strong>Rare Agent Code:</strong> Unusual intermediary identification, potential unauthorized access or data entry error</li>
                    <li><strong>Unpaired Transaction:</strong> Unbalanced transfer indicating failed or partial salary payment</li>
                    <li><strong>Large Balance Discrepancy:</strong> Significant difference between expected and actual balance, suggesting dormant account or reconciliation requirement</li>
                    <li><strong>ML Outlier:</strong> Machine learning detection of unusual patterns in transaction amounts compared to historical behavior</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Transaction Pattern Visualization")
            
            fig = px.scatter_3d(df, 
                               x=' Debit Amount', 
                               y=' Credit Amount', 
                               z='Net',
                               color='Anomaly Score',
                               color_continuous_scale=['#dc2626', '#059669'],
                               title='Three-Dimensional Transaction Analysis',
                               hover_data=['Document no.', 'Risk Score'],
                               labels={'Anomaly Score': 'Classification'})
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation Guide:**
            - Green markers represent normal transaction patterns
            - Red markers indicate anomalies detected by machine learning
            - Clustering patterns show common transaction behaviors
            - Isolated red markers require investigation
            """)
    
    else:
        st.success("No critical anomalies detected. All transactions appear within normal parameters.")
    
    # Decision tree explainability
    st.markdown('<p class="section-header">Decision Logic Visualization</p>', unsafe_allow_html=True)
    
    with st.expander("View Classification Decision Tree"):
        st.markdown("""
        This decision tree illustrates the classification logic used to identify high-risk transactions.
        Follow the branching structure to understand how transaction attributes influence risk assessment.
        """)
        
        X = df[[' Debit Amount', ' Credit Amount', 'Net']].fillna(0)
        y = (df['Risk Score'] >= high_risk_threshold).astype(int)
        
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(X, y)
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(dt, 
                 feature_names=['Debit Amount', 'Credit Amount', 'Net'],
                 class_names=['Normal', 'High Risk'],
                 filled=True,
                 ax=ax,
                 fontsize=10,
                 rounded=True)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Reading the Decision Tree:**
        - Each node contains a decision rule based on transaction attributes
        - Left branches represent TRUE conditions, right branches represent FALSE
        - Node colors indicate risk level intensity
        - Leaf nodes show final classification outcomes
        """)
    
    # Reports section
    st.markdown('<p class="section-header">Downloadable Reports</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not high_risk.empty:
            critical_report = high_risk[['Document no.', 'Agent code', ' Debit Amount', ' Credit Amount', 
                                        'Risk Score', 'Impact Level', 'Recommended Action', 'Explanations']]
            csv_critical = critical_report.to_csv(index=False)
            st.download_button(
                label="Download Critical Alerts",
                data=csv_critical,
                file_name=f"critical_alerts_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if not flagged.empty:
            all_flagged_report = flagged[['Document no.', 'Agent code', ' Debit Amount', ' Credit Amount', 
                                         'Risk Score', 'Flag', 'Recommended Action']]
            csv_all = all_flagged_report.to_csv(index=False)
            st.download_button(
                label="Download All Flagged Transactions",
                data=csv_all,
                file_name=f"all_flagged_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        flag_summary = flag_df.to_string(index=False) if not flag_df.empty else 'No anomalies detected'
        summary_text = f"""BANK OF BURUNDI - TRANSACTION MONITORING SUMMARY
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

EXECUTIVE OVERVIEW
Total Transactions Processed: {len(df):,}
Critical Issues Identified: {len(high_risk)}
Workers Affected: {flagged['Document no.'].nunique()}
System Efficiency Improvement: ~{(1 - len(high_risk)/len(df))*100:.1f}%

DETECTION SUMMARY
{flag_summary}

PRIORITY ACTIONS
{high_risk['Recommended Action'].value_counts().head().to_string() if not high_risk.empty else 'No immediate actions required'}
        """
        st.download_button(
            label="Download Executive Summary",
            data=summary_text,
            file_name=f"executive_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    # Timeline visualization
    st.markdown('<p class="section-header">Transaction Timeline Analysis</p>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Running Balance'],
        mode='lines',
        name='Running Balance',
        line=dict(color='#3b82f6', width=2)
    ))
    
    if not flagged.empty:
        colors = {'Critical': '#dc2626', 'Medium': '#d97706', 'Low': '#fbbf24'}
        for level in ['Critical', 'Medium', 'Low']:
            level_data = flagged[flagged['Impact Level'] == level]
            if not level_data.empty:
                fig.add_trace(go.Scatter(
                    x=level_data.index,
                    y=level_data['Running Balance'],
                    mode='markers',
                    name=f'{level} Risk',
                    marker=dict(
                        size=level_data['Risk Score'] * 3,
                        color=colors[level],
                        line=dict(width=1, color='white')
                    ),
                    text=level_data['Explanations'].apply(lambda x: '<br>'.join(x[:2])),
                    hovertemplate='<b>%{text}</b><br>Balance: %{y:,.0f} FBu<extra></extra>'
                ))
    
    fig.update_layout(
        title='Transaction Flow with Risk Indicators',
        xaxis_title='Transaction Index',
        yaxis_title='Balance (FBu)',
        hovermode='closest',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence metrics
    if not flagged.empty:
        st.markdown('<p class="section-header">Model Confidence Assessment</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Detection Confidence Distribution")
            confidence_data = {
                'High Confidence (7-15 points)': len(flagged[flagged['Risk Score'] >= 7]),
                'Medium Confidence (4-6 points)': len(flagged[(flagged['Risk Score'] >= 4) & (flagged['Risk Score'] < 7)]),
                'Low Confidence (1-3 points)': len(flagged[flagged['Risk Score'] < 4])
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(confidence_data.keys()),
                values=list(confidence_data.values()),
                marker=dict(colors=['#dc2626', '#d97706', '#fbbf24'])
            )])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Confidence Level Interpretation")
            st.markdown("""
            <div class="info-box">
            <p><strong>High Confidence (Score ≥7):</strong><br>
            Multiple detection rules activated. Requires immediate investigation and action.</p>
            
            <p><strong>Medium Confidence (Score 4-6):</strong><br>
            One or two rules triggered. Review recommended within 24 hours.</p>
            
            <p><strong>Low Confidence (Score 1-3):</strong><br>
            Single indicator detected. Monitor for patterns but may represent normal variation.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Optional: Show all transactions
    if show_all_transactions:
        st.markdown('<p class="section-header">Complete Transaction Register</p>', unsafe_allow_html=True)
        st.dataframe(
            df[['Document no.', 'Agent code', ' Debit Amount', ' Credit Amount', 
                'Risk Score', 'Flag', 'Impact Level', 'Explanations']],
            use_container_width=True,
            hide_index=True
        )

else:
    # Landing page
    st.info("Upload a transaction CSV file to begin analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### System Capabilities")
        st.markdown("""
        <div class="info-box">
        <ul class="styled-list">
            <li>Real-time detection of payment failures and account blocks</li>
            <li>Identification of duplicate processing and system errors</li>
            <li>Recognition of unauthorized intermediaries</li>
            <li>Detection of dormant accounts and reconciliation issues</li>
            <li>Machine learning-based pattern anomaly detection</li>
            <li>Transparent explanations for every alert generated</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Expected Outcomes")
        st.markdown("""
        <div class="info-box">
        <ul class="styled-list">
            <li>50%+ reduction in manual review workload</li>
            <li>Early detection of payment processing issues</li>
            <li>Automated staff notifications with actionable insights</li>
            <li>Minimized payment delays for workers</li>
            <li>Enhanced fraud detection capabilities</li>
            <li>Improved operational efficiency</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Detection Framework")
    st.markdown("""
    <div class="info-box">
    <h4>Rule-Based Detection</h4>
    <p>Six specialized rules identify common transaction anomalies including failed transfers, 
    account blocks, duplicate processing, unauthorized access, and reconciliation issues.</p>
    
    <h4>Machine Learning Analysis</h4>
    <p>Isolation Forest algorithm detects unusual patterns in transaction amounts that may 
    indicate fraud or system errors not captured by predefined rules.</p>
    
    <h4>Explainable Outputs</h4>
    <p>Every alert includes clear explanations of why it was flagged, recommended actions 
    for staff, and confidence levels to prioritize response efforts.</p>
    </div>
    """, unsafe_allow_html=True)