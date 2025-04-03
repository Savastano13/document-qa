import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pdfplumber
import requests
from datetime import datetime
from openai import OpenAI
import time
import re
import numpy as np

# ----------------- CONFIGURATION TELEGRAM -----------------
TELEGRAM_BOT_TOKEN = "7960645078:AAEzbayN_Kj1BV1rMRCg1LebvzDjPwFazYU"
TELEGRAM_CHAT_ID = "6672006206"

def send_telegram_message(message):
    """
    Envoie un message via Telegram pour notifier l'utilisateur des événements importants.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=payload)

# ----------------- CONFIGURATION STREAMLIT -----------------
st.set_page_config(page_title="IA Finance + Bot Trading BTC", layout="wide")
st.sidebar.title("🔐 Configuration & Authentification")
openai_api_key = st.sidebar.text_input("Clé API OpenAI (facultatif)", type="password")

# Mode test si aucune clé n'est fournie (les réponses IA seront simulées)
mode_test = not openai_api_key
if not mode_test:
    openai.api_key = openai_api_key
    client_openai = OpenAI(api_key=openai_api_key)
else:
    client_openai = None

# ----------------- ONGLETS PRINCIPAUX -----------------
tab1, tab2, tab3 = st.tabs(["💬 Conseiller Financier IA", "📈 Bot de Trading BTC/USDT", "📊 Analyse de Portefeuille"])

# ----------------- FONCTION UTILE : Extraction des Transactions -----------------
def extract_transactions(text):
    """
    Extraction approximative des transactions depuis un relevé bancaire.
    Cherche des lignes contenant une date (format dd/mm/yyyy) et un montant.
    """
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}).*?([-+]?\d+[.,]?\d*)"
    matches = re.findall(pattern, text)
    transactions = []
    for m in matches:
        date_str = m[0]
        amount = m[1].replace(',', '.')
        try:
            # On essaie d'interpréter la date sur 4 puis sur 2 chiffres
            try:
                date_obj = datetime.strptime(date_str, "%d/%m/%Y")
            except ValueError:
                date_obj = datetime.strptime(date_str, "%d/%m/%y")
        except Exception:
            date_obj = None
        transactions.append({"Date": date_obj, "Montant": float(amount)})
    df = pd.DataFrame(transactions)
    return df if not df.empty else None

# ===================== TAB 1 : Conseiller Financier IA =====================
with tab1:
    st.title("💬 Votre Conseiller Financier IA")
    st.markdown(
        """
        Bienvenue dans le module de Conseil Financier IA !  
        Ici, vous pouvez téléverser votre relevé bancaire (au format TXT ou PDF) et poser une question sur vos dépenses.  
        Notre IA analysera votre relevé et vous fournira des conseils personnalisés pour optimiser votre budget.
        
        **Instructions d'utilisation :**
        1. **Téléversez votre relevé bancaire** : Utilisez le sélecteur ci-dessous pour choisir un fichier TXT ou PDF.
        2. **Posez votre question** : Exprimez clairement votre interrogation (par exemple, "Où est-ce que je dépense trop ?").
        3. **Extraire les transactions (facultatif)** : Cliquez sur le bouton pour tenter d'extraire automatiquement les transactions présentes dans votre relevé.
        4. **Lancez l'analyse** : Appuyez sur le bouton pour démarrer l'analyse et obtenir une réponse détaillée de l'IA.
        5. **Historique** : Consultez l'historique des échanges affiché ci-dessous et téléchargez-le si besoin.
        """
    )
    
    # Téléversement du relevé bancaire et saisie de la question
    uploaded_file = st.file_uploader("📎 Téléversez votre relevé bancaire", type=["txt", "pdf"])
    question = st.text_area("Posez votre question :", placeholder="Ex : Où est-ce que je dépense trop ?")
    
    # Initialisation de l'historique des échanges (stocké dans la session)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if uploaded_file and question:
        try:
            # Lecture du document en fonction du format
            if uploaded_file.name.endswith(".txt"):
                document = uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith(".pdf"):
                with pdfplumber.open(uploaded_file) as pdf:
                    document = "\n".join([page.extract_text() or "" for page in pdf.pages])
            else:
                st.error("❌ Format non supporté.")
                st.stop()
            
            # Bouton d'extraction des transactions
            st.markdown("#### 🔎 Extraction facultative des transactions")
            if st.button("Extraire les transactions du relevé"):
                transactions_df = extract_transactions(document)
                if transactions_df is not None:
                    st.markdown("**Transactions détectées :**")
                    st.dataframe(transactions_df)
                else:
                    st.info("Aucune transaction n'a pu être détectée automatiquement.")
            
            st.markdown("#### 📢 Lancement de l'analyse du relevé bancaire")
            # Lancement de l'analyse
            if st.button("Lancer l'analyse du relevé"):
                if mode_test:
                    # Réponse simulée en mode test
                    fake_response = (
                        "Réponse simulée : Votre relevé montre des dépenses excessives en abonnements et frais de livraison. "
                        "Il serait judicieux de revoir ces postes afin d'optimiser votre budget mensuel."
                    )
                    st.success(fake_response)
                    st.session_state.chat_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "user": question,
                        "bot": fake_response
                    })
                    send_telegram_message(f"[TEST IA] {fake_response}")
                else:
                    # Construction du prompt pour l'IA
                    system_prompt = (
                        "Tu es un conseiller financier expert. Analyse le relevé bancaire fourni et donne des conseils concrets pour "
                        "optimiser les dépenses et économiser de l'argent. Fournis des recommandations pratiques et détaillées."
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Relevé bancaire :\n{document}\n\nQuestion : {question}"}
                    ]
                    with st.spinner("💬 L'IA analyse votre relevé..."):
                        response = client_openai.chat.completions.create(
                            model="gpt-4",
                            messages=messages,
                            stream=True,
                        )
                        full_response = ""
                        placeholder = st.empty()
                        for part in response:
                            content = part['choices'][0].get('delta', {}).get('content', '')
                            full_response += content
                            placeholder.markdown(full_response)
                        st.session_state.chat_history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "user": question,
                            "bot": full_response
                        })
                        send_telegram_message(f"📩 Résultat IA :\n{full_response[:400]}...")
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement : {e}")
    
    # Affichage de l'historique des échanges
    if st.session_state.chat_history:
        st.markdown("### Historique des échanges")
        history_df = pd.DataFrame(st.session_state.chat_history)
        st.dataframe(history_df)
        csv_history = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger l'historique", data=csv_history, file_name="historique_chat.csv", mime="text/csv")
        for exchange in st.session_state.chat_history:
            st.markdown(f"**[{exchange['timestamp']}] Vous :** {exchange['user']}")
            st.markdown(f"**[{exchange['timestamp']}] IA :** {exchange['bot']}")

with tab2:
    st.title("📈 Bot de Trading BTC/USDT")
    st.markdown(
        """
        **Bienvenue dans le Bot de Trading BTC/USDT !**

        Ce module vous permet d'analyser le marché du Bitcoin en temps réel en utilisant des indicateurs techniques avancés.
        Vous pouvez personnaliser les paramètres d'analyse pour adapter l'outil à votre stratégie de trading.

        **Comment utiliser ce module :**
        1. **Sélection de la période d'analyse :**
           - Choisissez la durée (en jours) sur laquelle vous souhaitez analyser les données (1, 7 ou 30 jours).
           - Cette période définit l'historique des données OHLC (Open, High, Low, Close) récupérées depuis CoinGecko.
        2. **Paramètres personnalisés :**
           - **Seuil RSI :** Utilisez le curseur pour définir le seuil du Relative Strength Index (RSI).
             Par défaut, 50 est utilisé, ce qui est souvent le point de bascule entre tendance haussière et baissière.
           - **Décalage MACD :** Entrez un décalage (offset) pour ajuster la sensibilité du MACD (Moving Average Convergence Divergence).
             La valeur par défaut est 0, mais vous pouvez l'ajuster pour affiner l'analyse.
        3. **Lancement de l'analyse :**
           - Cliquez sur le bouton **"🚀 Lancer l'analyse de trading"** pour lancer l'analyse du marché.
           - Le bot récupérera les données, calculera les indicateurs (EMA, RSI, MACD, Bollinger Bands), et affichera des graphiques détaillés.
        4. **Interprétation des résultats :**
           - Vous verrez un graphique présentant le prix du BTC avec les indicateurs EMA, Bollinger Bands, RSI et MACD.
           - Un signal sera généré : **BUY** si les conditions haussières sont remplies, **SELL** si les conditions baissières sont détectées, sinon **NEUTRAL**.
           - En cas de signal BUY ou SELL, une notification sera envoyée via Telegram.
        """
    )

    # Sélection de la période d'analyse pour les données OHLC
    days = st.selectbox("Sélectionnez la période d'analyse (en jours)", options=[1, 7, 30], index=0)

    # Paramètres personnalisables pour l'analyse technique
    st.markdown("#### Personnalisez vos paramètres d'analyse")
    rsi_threshold = st.slider("Seuil RSI (défaut 50)", 30, 70, 50)
    macd_offset = st.number_input("Décalage minimal MACD (défaut 0)", value=0.0, step=0.1)

    @st.cache_data
    def get_btc_price():
        """Récupère le prix actuel de BTC/USDT depuis CoinGecko."""
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data["bitcoin"]["usd"]
        except Exception as e:
            st.error(f"❌ Erreur lors de la récupération du prix BTC/USDT: {e}")
            st.stop()

    @st.cache_data
    def get_btc_ohlc(days):
        """Récupère les données OHLC de BTC/USDT depuis CoinGecko pour la période sélectionnée."""
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days={days}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("time", inplace=True)
            return df
        except Exception as e:
            st.error(f"❌ Erreur lors de la récupération des données OHLC: {e}")
            st.stop()

    def calculate_indicators(df):
        """Calcule les indicateurs techniques : EMA, RSI, MACD et Bollinger Bands."""
        df["EMA50"] = df["close"].ewm(span=50).mean()
        df["EMA200"] = df["close"].ewm(span=200).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean() + macd_offset
        # Calcul des Bollinger Bands sur une moyenne mobile sur 20 périodes
        df["MA20"] = df["close"].rolling(window=20).mean()
        df["BB_upper"] = df["MA20"] + 2 * df["close"].rolling(window=20).std()
        df["BB_lower"] = df["MA20"] - 2 * df["close"].rolling(window=20).std()
        return df

    def analyse_trading(df):
        """Détermine un signal de trading (BUY/SELL/NEUTRAL) en fonction des indicateurs techniques."""
        last = df.iloc[-1]
        if last["close"] > last["EMA50"] > last["EMA200"] and last["RSI"] > rsi_threshold and last["MACD"] > last["MACD_Signal"]:
            return "BUY", "Signal d'achat détecté : tendance haussière confirmée par les indicateurs."
        elif last["close"] < last["EMA50"] < last["EMA200"] and last["RSI"] < rsi_threshold and last["MACD"] < last["MACD_Signal"]:
            return "SELL", "Signal de vente détecté : tendance baissière confirmée par les indicateurs."
        else:
            return "NEUTRAL", "Aucun signal clair détecté actuellement."

    def plot_trading_chart(df):
        """Affiche un graphique détaillé du prix avec EMA, Bollinger Bands, RSI et MACD."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # Graphique du prix avec EMA et Bollinger Bands
        ax1.plot(df.index, df["close"], label="Prix BTC", color="black")
        ax1.plot(df.index, df["EMA50"], label="EMA 50", color="blue")
        ax1.plot(df.index, df["EMA200"], label="EMA 200", color="red")
        ax1.plot(df.index, df["BB_upper"], label="Bollinger Upper", linestyle="--", color="green")
        ax1.plot(df.index, df["BB_lower"], label="Bollinger Lower", linestyle="--", color="green")
        ax1.fill_between(df.index, df["BB_lower"], df["BB_upper"], color="green", alpha=0.1)
        ax1.set_title("Cours BTC/USDT et indicateurs techniques")
        ax1.legend()
        # Graphique des oscillateurs RSI et MACD
        ax2.plot(df.index, df["RSI"], label="RSI", color="purple")
        ax2.plot(df.index, df["MACD"], label="MACD", color="orange")
        ax2.plot(df.index, df["MACD_Signal"], label="Signal MACD", linestyle="--", color="gray")
        ax2.set_title("RSI et MACD")
        ax2.legend()
        st.pyplot(fig)

    # Affichage du prix actuel de BTC/USDT
    current_price = get_btc_price()
    st.metric("💰 Prix actuel BTC/USDT", f"${current_price:.2f}")

    # Bouton pour lancer l'analyse de trading
    if st.button("🚀 Lancer l'analyse de trading"):
        df_btc = get_btc_ohlc(days)
        df_btc = calculate_indicators(df_btc)
        signal, comment = analyse_trading(df_btc)
        plot_trading_chart(df_btc)
        st.markdown("### Résultat de l'analyse de trading")
        st.write(f"**Signal détecté :** {signal}")
        st.write(f"**Commentaire :** {comment}")
        if signal != "NEUTRAL":
            send_telegram_message(f"📈 Signal Trading: {signal}\n{comment}")
            st.success(f"✅ Signal envoyé par Telegram : {signal}")
        else:
            send_telegram_message("🔍 Aucun signal clair détecté.")
            st.info("Aucune opportunité détectée pour le moment.")

    with st.expander("ℹ️ Explications sur les indicateurs utilisés"):
        st.markdown(
            """
            - **EMA 50 / EMA 200** : Moyennes mobiles qui indiquent la tendance à court et à long terme.
              - **EMA 50** : Moyenne mobile exponentielle sur 50 périodes, réactive aux changements de prix récents.
              - **EMA 200** : Moyenne mobile exponentielle sur 200 périodes, indicateur de tendance à long terme.
            - **RSI** : Indice de force relative mesurant le momentum; un RSI supérieur à 50 suggère une tendance haussière.
              - **Seuil RSI** : Un RSI élevé (>70) peut indiquer une surachat, tandis qu'un RSI bas (<30) peut indiquer une survente.
            - **MACD** : Indicateur de tendance calculé à partir de la différence entre deux moyennes mobiles; le décalage permet d'ajuster la sensibilité.
              - **MACD Line** : Différence entre les moyennes mobiles exponentielles de 12 et 26 périodes.
              - **Signal Line** : Moyenne mobile exponentielle de la MACD Line sur 9 périodes.
            - **Bollinger Bands** : Indicateurs de volatilité basés sur une moyenne mobile sur 20 périodes et ±2 écarts-types.
              - **Bandes supérieure et inférieure** : Indiquent les niveaux de support et de résistance dynamiques.
            """
        )
# ===================== TAB 3 : Analyse de Portefeuille =====================
with tab3:
    st.title("📊 Analyse de Portefeuille - Optimisez vos Investissements")
    st.markdown(
        "Téléversez votre portefeuille au format CSV pour obtenir une analyse complète : répartition des actifs, valeur totale, indice de diversification, et même une estimation de la volatilité globale. "
        "Vous pourrez aussi poser une question à l'IA pour des recommandations personnalisées et télécharger le rapport complet."
    )
    uploaded_portfolio = st.file_uploader("Téléversez votre portefeuille (CSV: Asset, Quantity, Price)", type=["csv"])
    if uploaded_portfolio is not None:
        portfolio_df = pd.read_csv(uploaded_portfolio)
    else:
        # Portefeuille exemple
        data = {'Asset': ['BTC', 'ETH', 'AAPL', 'GOOGL', 'AMZN'],
                'Quantity': [0.5, 2, 10, 5, 8],
                'Price': [60000, 4000, 150, 2800, 3500]}
        portfolio_df = pd.DataFrame(data)
        st.info("Utilisation d'un portefeuille exemple.")
    
    # Calcul de la valeur par actif et de la répartition
    portfolio_df["Value"] = portfolio_df["Quantity"] * portfolio_df["Price"]
    total_value = portfolio_df["Value"].sum()
    portfolio_df["Weight (%)"] = portfolio_df["Value"] / total_value * 100
    st.subheader("Détails du Portefeuille")
    st.dataframe(portfolio_df)
    st.metric("Valeur Totale", f"${total_value:,.2f}")
    
    # Calcul de l'indice de diversification (Indice de Herfindahl-Hirschman)
    portfolio_df["Weight_Fraction"] = portfolio_df["Weight (%)"] / 100
    herfindahl_index = (portfolio_df["Weight_Fraction"] ** 2).sum()
    diversification_index = 1 - herfindahl_index
    st.write(f"**Indice de Diversification :** {diversification_index:.2f} (0 = faible, 1 = forte diversification)")
    
    # Estimation de la volatilité du portefeuille (écart-type des valeurs)
    volatility = portfolio_df["Value"].std()
    st.write(f"**Volatilité estimée du portefeuille :** {volatility:.2f}")
    
    # Graphique circulaire de répartition
    fig_port, ax_port = plt.subplots(figsize=(6,6))
    ax_port.pie(portfolio_df["Value"], labels=portfolio_df["Asset"], autopct='%1.1f%%', startangle=90)
    ax_port.axis('equal')
    st.pyplot(fig_port)
    
    # Graphique à barres des valeurs par actif
    fig_bar, ax_bar = plt.subplots(figsize=(8,4))
    ax_bar.bar(portfolio_df["Asset"], portfolio_df["Value"], color="skyblue")
    ax_bar.set_title("Valeur par Actif")
    ax_bar.set_ylabel("Valeur ($)")
    st.pyplot(fig_bar)
    
    # Option de téléchargement du rapport complet
    csv_report = portfolio_df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger le rapport CSV", data=csv_report, file_name="rapport_portefeuille.csv", mime="text/csv")
    
    st.markdown("### Consultation IA pour recommandations")
    portfolio_question = st.text_area("Posez votre question sur votre portefeuille :", placeholder="Ex : Comment optimiser ma diversification et réduire le risque ?")
    if portfolio_question:
        portfolio_summary = portfolio_df.to_csv(index=False)
        if mode_test:
            fake_response = ("Analyse simulée : Votre portefeuille est globalement équilibré, mais vous pourriez augmenter l'exposition aux actions technologiques pour améliorer la performance à long terme tout en surveillant le risque global.")
            st.success(fake_response)
            send_telegram_message(f"[TEST Portefeuille] {fake_response}")
        else:
            system_prompt = (
                "Tu es un expert en gestion de portefeuille et en finances personnelles. "
                "Analyse le portefeuille fourni et propose des recommandations pour optimiser la diversification, réduire le risque et améliorer la performance."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Voici mon portefeuille au format CSV :\n{portfolio_summary}\n\nQuestion : {portfolio_question}"}
            ]
            with st.spinner("💬 L'IA analyse votre portefeuille..."):
                response = client_openai.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    stream=True,
                )
                portfolio_response = ""
                placeholder_portfolio = st.empty()
                for part in response:
                    content = part['choices'][0].get('delta', {}).get('content', '')
                    portfolio_response += content
                    placeholder_portfolio.markdown(portfolio_response)
                send_telegram_message(f"📩 Analyse Portefeuille IA :\n{portfolio_response[:400]}...")

# ----------------- FIN DU CODE -----------------
