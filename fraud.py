import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, plot_tree
import io
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Intelligence Artificiel- Système de Surveillance des Transactions", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style professionnel
st.markdown("""
<style>
    /* Couleurs principales du thème */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --success-color: #059669;
        --warning-color: #d97706;
        --danger-color: #dc2626;
        --neutral-bg: #f8fafc;
    }
    
    /* Supprimer le style streamlit par défaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Style de l'en-tête */
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
    
    /* Cartes de niveau de risque */
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
    
    /* Boîtes d'information */
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
    
    /* Cartes métriques */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Détails de transaction */
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
    
    /* Boutons d'action */
    .action-required {
        background-color: #f0f9ff;
        border: 1px solid #7dd3fc;
        padding: 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #0c4a6e;
    }
    
    /* En-têtes de section */
    .section-header {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Style de liste */
    .styled-list {
        color: #475569;
        line-height: 1.8;
    }
    
    .styled-list li {
        margin-bottom: 0.5rem;
    }
    
    /* Style de tableau */
    .dataframe {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown("""
<div class="main-header">
    <h1>Intelligence Artificiel- Système de Surveillance des Transactions</h1>
    <p>Détection intelligente d'anomalies pour les opérations de décaissement des salaires</p>
</div>
""", unsafe_allow_html=True)

# Configuration de la barre latérale
with st.sidebar:
    st.markdown("### Configuration du Système")
    
    st.markdown("#### Catégories de risques")
    high_risk_threshold = st.slider("Risque Élevé", 3, 10, 7, help="Les transactions dépassant ce seuil nécessitent une attention immédiate")
    medium_risk_threshold = st.slider("Risque Moyen", 2, 7, 4, help="Les transactions dépassant ce seuil nécessitent un examen")
    
    st.markdown("#### Sensibilité de Détection")
    contamination = st.slider("Taux de Détection d'Anomalies (%)", 1, 20, 5, help="Pourcentage attendu de transactions anormales") / 100
    
    st.markdown("#### Options d'Affichage")
    show_all_transactions = st.checkbox("Afficher Toutes les Transactions", False)
    enable_auto_alerts = st.checkbox("Activer les Alertes Automatiques", False)
    
    st.markdown("---")
    st.markdown("### Guide Rapide")
    st.markdown("""
    1. Télécharger le fichier CSV des transactions
    2. Examiner les anomalies signalées
    3. Télécharger les rapports d'action
    4. Configurer les préférences d'alerte
    """)

# Téléchargement de fichier
uploaded_file = st.file_uploader(
    "Télécharger les Données de Transaction", 
    type="csv", 
    help="Sélectionner le fichier CSV de décaissement des salaires de la Banque de la République du Burundi"
)

if uploaded_file is not None:
    # Charger et traiter les données
    df = pd.read_csv(uploaded_file)
    
    # Prétraitement
    numeric_cols = [' Debit Amount', ' Credit Amount', ' Account balance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Ingénierie des caractéristiques
    df['Net'] = df[' Credit Amount'] - df[' Debit Amount']
    df['Running Balance'] = df['Net'].cumsum()
    df['Flag'] = ''
    df['Risk Score'] = 0
    df['Explanations'] = [[] for _ in range(len(df))]
    df['Recommended Action'] = ''
    df['Impact Level'] = 'Faible'
    
    # Règle 1: Documents Hautement Dupliqués
    doc_counts = df['Document no.'].value_counts()
    high_duplicates = doc_counts[doc_counts > 3].index
    mask = df['Document no.'].isin(high_duplicates)
    df.loc[mask, 'Flag'] += 'Nombre de Duplications Élevé; '
    df.loc[mask, 'Risk Score'] += 4
    for idx in df[mask].index:
        count = doc_counts[df.loc[idx, 'Document no.']]
        df.at[idx, 'Explanations'].append(f"Le document apparaît {count} fois (seuil: 3). Erreur potentielle de traitement par lot ou nouvelle tentative système détectée.")
        df.at[idx, 'Recommended Action'] = 'Vérifier la légitimité de multiples crédits salariaux. Examiner les enregistrements d\'employés en double.'
    
    # Règle 2: Net Négatif
    mask = df['Net'] < 0
    df.loc[mask, 'Flag'] += 'Net Négatif; '
    df.loc[mask, 'Risk Score'] += 3
    for idx in df[mask].index:
        debit = df.loc[idx, ' Debit Amount']
        credit = df.loc[idx, ' Credit Amount']
        df.at[idx, 'Explanations'].append(f"Le montant débit ({debit:,.0f} FBu) dépasse le montant crédit ({credit:,.0f} FBu). Le compte du travailleur peut avoir des obligations de prêt en cours ou des blocages actifs.")
        df.at[idx, 'Recommended Action'] = 'Contacter le titulaire du compte pour vérifier le statut du prêt et les restrictions du compte.'
        df.at[idx, 'Impact Level'] = 'Élevé'
    
    # Règle 3: Code Agent Rare
    agent_counts = df['Agent code'].value_counts()
    rare_agents = agent_counts[agent_counts <= 1].index
    mask = df['Agent code'].isin(rare_agents)
    df.loc[mask, 'Flag'] += 'Code Agent Rare; '
    df.loc[mask, 'Risk Score'] += 2
    for idx in df[mask].index:
        agent = df.loc[idx, 'Agent code']
        df.at[idx, 'Explanations'].append(f"Le code agent '{agent}' n'apparaît qu'une seule fois dans l'ensemble de données. Peut indiquer un intermédiaire non autorisé ou une erreur de saisie de données.")
        df.at[idx, 'Recommended Action'] = 'Vérifier l\'autorisation de l\'agent par rapport au registre approuvé.'
    
    # Règle 4: Transactions Non Appariées
    grouped_net = df.groupby('Document no.')['Net'].sum()
    unpaired_docs = grouped_net[grouped_net != 0].index
    mask = df['Document no.'].isin(unpaired_docs)
    df.loc[mask, 'Flag'] += 'Transaction Non Appariée; '
    df.loc[mask, 'Risk Score'] += 5
    for idx in df[mask].index:
        doc = df.loc[idx, 'Document no.']
        net_sum = grouped_net[doc]
        df.at[idx, 'Explanations'].append(f"Transaction non équilibrée (somme nette: {net_sum:,.0f} FBu). Le transfert a probablement échoué ou a été partiellement traité.")
        df.at[idx, 'Recommended Action'] = 'Enquête prioritaire requise. Le travailleur n\'a peut-être pas reçu le paiement du salaire.'
        df.at[idx, 'Impact Level'] = 'Critique'
    
    # Règle 5: Écart de Solde Important
    df['Discrepancy'] = abs(df['Running Balance'] - df[' Account balance'])
    large_disc_threshold = df[' Debit Amount'].mean() * 2
    mask = df['Discrepancy'] > large_disc_threshold
    df.loc[mask, 'Flag'] += 'Écart de Solde Important; '
    df.loc[mask, 'Risk Score'] += 4
    for idx in df[mask].index:
        disc = df.loc[idx, 'Discrepancy']
        df.at[idx, 'Explanations'].append(f"Écart de solde de {disc:,.0f} FBu détecté (seuil: {large_disc_threshold:,.0f} FBu). Le compte peut être inactif ou nécessiter un rapprochement.")
        df.at[idx, 'Recommended Action'] = 'Examiner l\'historique d\'activité du compte et rapprocher les enregistrements de solde.'
        df.at[idx, 'Impact Level'] = 'Moyen'
    
    # Détection d'Anomalies par Apprentissage Automatique
    features = df[[' Debit Amount', ' Credit Amount', 'Net']].copy()
    iso = IsolationForest(contamination=contamination, random_state=42)
    df['Anomaly Score'] = iso.fit_predict(features)
    df['Anomaly Probability'] = iso.score_samples(features)
    
    mask = df['Anomaly Score'] == -1
    df.loc[mask, 'Flag'] += 'Valeur Aberrante ML; '
    df.loc[mask, 'Risk Score'] += 3
    
    for idx in df[mask].index:
        debit = df.loc[idx, ' Debit Amount']
        credit = df.loc[idx, ' Credit Amount']
        
        debit_mean, debit_std = df[' Debit Amount'].mean(), df[' Debit Amount'].std()
        credit_mean, credit_std = df[' Credit Amount'].mean(), df[' Credit Amount'].std()
        
        contributors = []
        if abs(debit - debit_mean) > 2 * debit_std:
            contributors.append(f"Le montant débit ({debit:,.0f} FBu) s'écarte de {abs(debit - debit_mean)/debit_std:.1f} écarts types de la moyenne")
        if abs(credit - credit_mean) > 2 * credit_std:
            contributors.append(f"Le montant crédit ({credit:,.0f} FBu) s'écarte de {abs(credit - credit_mean)/credit_std:.1f} écarts types de la moyenne")
        
        explanation = "Le modèle d'apprentissage automatique a détecté un schéma inhabituel: " + ("; ".join(contributors) if contributors else "Les montants des transactions montrent un écart significatif par rapport aux normes historiques")
        df.at[idx, 'Explanations'].append(explanation)
        if not df.at[idx, 'Recommended Action']:
            df.at[idx, 'Recommended Action'] = 'Examiner la transaction pour détecter d\'éventuels indicateurs de fraude ou erreurs de données.'
    
    # Catégorisation des risques
    df.loc[df['Risk Score'] >= high_risk_threshold, 'Impact Level'] = 'Critique'
    df.loc[(df['Risk Score'] >= medium_risk_threshold) & (df['Risk Score'] < high_risk_threshold), 'Impact Level'] = 'Moyen'
    
    # Filtrer les transactions signalées
    flagged = df[df['Flag'] != ''].copy()
    high_risk = flagged[flagged['Risk Score'] >= high_risk_threshold].sort_values('Risk Score', ascending=False)
    
    # Métriques du tableau de bord
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total des Transactions", f"{len(df):,}")
    with col2:
        critical_count = len(flagged[flagged['Impact Level'] == 'Critique'])
        st.metric("Problèmes Critiques", critical_count, delta=f"{critical_count/len(df)*100:.1f}%" if len(df) > 0 else "0%", delta_color="inverse")
    with col3:
        workers_affected = flagged['Document no.'].nunique()
        st.metric("Travailleurs Affectés", workers_affected)
    with col4:
        potential_loss = flagged[flagged['Net'] < 0]['Net'].sum()
        st.metric("Perte Potentielle", f"{abs(potential_loss):,.0f} FBu")
    
    # Section des alertes
    st.markdown('<p class="section-header">Résultats de Détection et Actions Recommandées</p>', unsafe_allow_html=True)
    
    if not high_risk.empty:
        tab1, tab2, tab3 = st.tabs(["Alertes Critiques", "Performance des Règles", "Analyse des Schémas"])
        
        with tab1:
            st.markdown("### Transactions Hautement Prioritaires")
            
            for idx, row in high_risk.head(10).iterrows():
                risk_class = "risk-critical" if row['Impact Level'] == 'Critique' else "risk-medium"
                
                st.markdown(f"""
                <div class="{risk_class}">
                    <div class="transaction-header">Transaction #{idx} | Document: {row['Document no.']} | Score de Risque: {row['Risk Score']}/15</div>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Raisonnement de Détection:**")
                    for explanation in row['Explanations']:
                        st.markdown(f'<div class="explanation-item">{explanation}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Action Recommandée:**")
                    st.markdown(f'<div class="action-required">{row["Recommended Action"]}</div>', unsafe_allow_html=True)
                    st.markdown(f"**Niveau d'Impact:** `{row['Impact Level']}`")
                    st.markdown(f"**Débit:** {row[' Debit Amount']:,.0f} FBu | **Crédit:** {row[' Credit Amount']:,.0f} FBu")
                
                st.markdown("---")
        
        with tab2:
            st.markdown("### Performance des Règles de Détection")
            
            flag_analysis = []
            for flag_type in ['Nombre de Duplications Élevé', 'Net Négatif', 'Code Agent Rare', 'Transaction Non Appariée', 'Écart de Solde Important', 'Valeur Aberrante ML']:
                count = flagged['Flag'].str.contains(flag_type).sum()
                if count > 0:
                    flag_analysis.append({
                        'Règle de Détection': flag_type,
                        'Occurrences': count,
                        'Pourcentage': f"{count/len(df)*100:.2f}%"
                    })
            
            flag_df = pd.DataFrame(flag_analysis)
            
            if not flag_df.empty:
                fig = px.bar(flag_df, x='Règle de Détection', y='Occurrences', 
                            title='Fréquence d\'Activation des Règles',
                            color='Occurrences',
                            color_continuous_scale='Blues')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(flag_df, use_container_width=True, hide_index=True)
                
                st.markdown("""
                <div class="info-box">
                <h4>Descriptions des Règles de Détection</h4>
                <ul class="styled-list">
                    <li><strong>Nombre de Duplications Élevé:</strong> Plusieurs transactions avec des numéros de document identiques suggérant des erreurs de traitement par lot ou des nouvelles tentatives système</li>
                    <li><strong>Net Négatif:</strong> Le débit dépasse le crédit, indiquant des déductions de prêt ou des restrictions de compte empêchant le crédit salarial complet</li>
                    <li><strong>Code Agent Rare:</strong> Identification d'intermédiaire inhabituelle, accès potentiellement non autorisé ou erreur de saisie de données</li>
                    <li><strong>Transaction Non Appariée:</strong> Transfert non équilibré indiquant un paiement de salaire échoué ou partiel</li>
                    <li><strong>Écart de Solde Important:</strong> Différence significative entre le solde attendu et réel, suggérant un compte inactif ou une exigence de rapprochement</li>
                    <li><strong>Valeur Aberrante ML:</strong> Détection par apprentissage automatique de schémas inhabituels dans les montants de transaction par rapport au comportement historique</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Visualisation des Schémas de Transaction")
            
            fig = px.scatter_3d(df, 
                               x=' Debit Amount', 
                               y=' Credit Amount', 
                               z='Net',
                               color='Anomaly Score',
                               color_continuous_scale=['#dc2626', '#059669'],
                               title='Analyse Tridimensionnelle des Transactions',
                               hover_data=['Document no.', 'Risk Score'],
                               labels={'Anomaly Score': 'Classification', ' Debit Amount': 'Montant Débit', ' Credit Amount': 'Montant Crédit'})
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Guide d'Interprétation:**
            - Les marqueurs verts représentent des schémas de transaction normaux
            - Les marqueurs rouges indiquent des anomalies détectées par apprentissage automatique
            - Les motifs de regroupement montrent des comportements de transaction courants
            - Les marqueurs rouges isolés nécessitent une enquête
            """)
    
    else:
        st.success("Aucune anomalie critique détectée. Toutes les transactions semblent dans les paramètres normaux.")
    
    # Explicabilité de l'arbre de décision
    st.markdown('<p class="section-header">Visualisation de la Logique de Décision</p>', unsafe_allow_html=True)
    
    with st.expander("Voir l'Arbre de Décision de Classification"):
        st.markdown("""
        Cet arbre de décision illustre la logique de classification utilisée pour identifier les transactions à haut risque.
        Suivez la structure de branchement pour comprendre comment les attributs de transaction influencent l'évaluation des risques.
        """)
        
        X = df[[' Debit Amount', ' Credit Amount', 'Net']].fillna(0)
        y = (df['Risk Score'] >= high_risk_threshold).astype(int)
        
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(X, y)
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(dt, 
                 feature_names=['Montant Débit', 'Montant Crédit', 'Net'],
                 class_names=['Normal', 'Risque Élevé'],
                 filled=True,
                 ax=ax,
                 fontsize=10,
                 rounded=True)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Lecture de l'Arbre de Décision:**
        - Chaque nœud contient une règle de décision basée sur les attributs de transaction
        - Les branches de gauche représentent les conditions VRAIES, les branches de droite représentent les conditions FAUSSES
        - Les couleurs des nœuds indiquent l'intensité du niveau de risque
        - Les nœuds feuilles montrent les résultats de classification finaux
        """)
    
    # Section des rapports
    st.markdown('<p class="section-header">Rapports Téléchargeables</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not high_risk.empty:
            critical_report = high_risk[['Document no.', 'Agent code', ' Debit Amount', ' Credit Amount', 
                                        'Risk Score', 'Impact Level', 'Recommended Action', 'Explanations']]
            csv_critical = critical_report.to_csv(index=False)
            st.download_button(
                label="Télécharger Alertes Critiques",
                data=csv_critical,
                file_name=f"alertes_critiques_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if not flagged.empty:
            all_flagged_report = flagged[['Document no.', 'Agent code', ' Debit Amount', ' Credit Amount', 
                                         'Risk Score', 'Flag', 'Recommended Action']]
            csv_all = all_flagged_report.to_csv(index=False)
            st.download_button(
                label="Télécharger Toutes Transactions Signalées",
                data=csv_all,
                file_name=f"toutes_signalees_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        flag_summary = flag_df.to_string(index=False) if not flag_df.empty else 'Aucune anomalie détectée'
        summary_text = f"""BANQUE DE LA RÉPUBLIQUE DU BURUNDI - RÉSUMÉ DE SURVEILLANCE DES TRANSACTIONS
Généré le: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

APERÇU GÉNÉRAL
Total des Transactions Traitées: {len(df):,}
Problèmes Critiques Identifiés: {len(high_risk)}
Travailleurs Affectés: {flagged['Document no.'].nunique()}
Amélioration de l'Efficacité du Système: ~{(1 - len(high_risk)/len(df))*100:.1f}%

RÉSUMÉ DE DÉTECTION
{flag_summary}

ACTIONS PRIORITAIRES
{high_risk['Recommended Action'].value_counts().head().to_string() if not high_risk.empty else 'Aucune action immédiate requise'}
        """
        st.download_button(
            label="Télécharger Résumé Exécutif",
            data=summary_text,
            file_name=f"resume_executif_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    # Visualisation de la chronologie
    st.markdown('<p class="section-header">Analyse de la Chronologie des Transactions</p>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Running Balance'],
        mode='lines',
        name='Solde Cumulé',
        line=dict(color='#3b82f6', width=2)
    ))
    
    if not flagged.empty:
        colors = {'Critique': '#dc2626', 'Moyen': '#d97706', 'Faible': '#fbbf24'}
        for level in ['Critique', 'Moyen', 'Faible']:
            level_data = flagged[flagged['Impact Level'] == level]
            if not level_data.empty:
                fig.add_trace(go.Scatter(
                    x=level_data.index,
                    y=level_data['Running Balance'],
                    mode='markers',
                    name=f'Risque {level}',
                    marker=dict(
                        size=level_data['Risk Score'] * 3,
                        color=colors[level],
                        line=dict(width=1, color='white')
                    ),
                    text=level_data['Explanations'].apply(lambda x: '<br>'.join(x[:2])),
                    hovertemplate='<b>%{text}</b><br>Solde: %{y:,.0f} FBu<extra></extra>'
                ))
    
    fig.update_layout(
        title='Flux de Transactions avec Indicateurs de Risque',
        xaxis_title='Index de Transaction',
        yaxis_title='Solde (FBu)',
        hovermode='closest',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Métriques de confiance
    if not flagged.empty:
        st.markdown('<p class="section-header">Évaluation de la Confiance du Modèle</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Distribution de la Confiance de Détection")
            confidence_data = {
                'Haute Confiance (7-15 points)': len(flagged[flagged['Risk Score'] >= 7]),
                'Confiance Moyenne (4-6 points)': len(flagged[(flagged['Risk Score'] >= 4) & (flagged['Risk Score'] < 7)]),
                'Faible Confiance (1-3 points)': len(flagged[flagged['Risk Score'] < 4])
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(confidence_data.keys()),
                values=list(confidence_data.values()),
                marker=dict(colors=['#dc2626', '#d97706', '#fbbf24'])
            )])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Interprétation du Niveau de Confiance")
            st.markdown("""
            <div class="info-box">
            <p><strong>Haute Confiance (Score ≥7):</strong><br>
            Plusieurs règles de détection activées. Nécessite une enquête et une action immédiates.</p>
            
            <p><strong>Confiance Moyenne (Score 4-6):</strong><br>
            Une ou deux règles activées. Examen requis pour confirmer le risque.</p>
            
            <p><strong>Faible Confiance (Score 1-3):</strong><br>
            Faible signal de détection. À considérer en conjonction avec d'autres indicateurs.</p>
            </div>
            """, unsafe_allow_html=True)