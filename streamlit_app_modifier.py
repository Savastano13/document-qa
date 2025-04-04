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

# ================= CONFIGURATION DE LA CLÉ API =================
# ATTENTION : Votre clé API est ici en clair, ce qui peut être risqué en production.
# Il est recommandé d'utiliser des variables d'environnement ou un gestionnaire de secrets.
OPENAI_API_KEY = "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"

# ================= CONFIGURATION TELEGRAM =================
TELEGRAM_BOT_TOKEN = "7960645078:AAEzbayN_Kj1BV1rMRCg1LebvzDjPwFazYU"
TELEGRAM_CHAT_ID = "6672006206"

def send_telegram_message(message):
    """
    Envoie un message via Telegram pour notifier l'utilisateur des événements importants.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Erreur lors de l'envoi du message Telegram : {e}")

# ================= CONFIGURATION STREAMLIT =================
# Configuration générale de la page
st.set_page_config(page_title="IA Finance + Bot Trading BTC", layout="wide")

# Titre de la sidebar avec des instructions
st.sidebar.title("🔐 Configuration & Authentification")
st.sidebar.info("La clé API OpenAI est intégrée directement dans le code.")

# Utilisation directe de la clé API intégrée
openai_api_key = OPENAI_API_KEY

# Mode test : si la clé API n'est pas renseignée (ici, toujours False)
mode_test = not openai_api_key
if not mode_test:
    # Configuration du client OpenAI avec la clé intégrée
    client_openai = OpenAI(api_key=openai_api_key)
else:
    client_openai = None

# ================= DÉFINITION DES ONGLETS PRINCIPAUX =================
tab1, tab2, tab3 = st.tabs(["💬 Conseiller Financier IA", "📈 Bot de Trading BTC/USDT", "📊 Analyse de Portefeuille"])

# ================= FONCTION UTILE : Extraction des Transactions =================
def extract_transactions(text):
    """
    Extraction approximative des transactions depuis un relevé bancaire.
    Recherche des lignes contenant une date (format dd/mm/yyyy ou dd/mm/yy) et un montant.
    Retourne un DataFrame avec la date et le montant de chaque transaction détectée.
    """
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}).*?([-+]?\d+[.,]?\d*)"
    matches = re.findall(pattern, text)
    transactions = []
    for m in matches:
        date_str = m[0]
        amount = m[1].replace(',', '.')
        try:
            # Essai sur format 4 chiffres puis sur 2 chiffres
            try:
                date_obj = datetime.strptime(date_str, "%d/%m/%Y")
            except ValueError:
                date_obj = datetime.strptime(date_str, "%d/%m/%y")
        except Exception:
            date_obj = None
        transactions.append({"Date": date_obj, "Montant": float(amount)})
    df = pd.DataFrame(transactions)
    return df if not df.empty else None

# ================= TAB 1 : Conseiller Financier IA =================
# ================= TAB 1 : Conseiller Financier IA =================
# ================= TAB 1 : Conseiller Financier IA =================
with tab1:
    st.title("💬 Votre Conseiller Financier IA")
    st.markdown(
        """
        **Bienvenue dans le module de Conseil Financier IA !**

        - Téléversez votre relevé bancaire (format TXT ou PDF).
        - Posez votre question sur vos dépenses (ex. : *Où est-ce que je dépense trop ?*).
        - Extraire automatiquement les transactions (optionnel).
        - Lancez l'analyse pour obtenir des conseils personnalisés.
        - Consultez et téléchargez l'historique de vos échanges.
        """
    )

    # Téléversement du fichier et saisie de la question
    uploaded_file = st.file_uploader("📎 Téléversez votre relevé bancaire", type=["txt", "pdf"])
    question = st.text_area("Posez votre question :", placeholder="Ex : Où est-ce que je dépense trop ?")

    # Initialisation de l'historique des échanges dans la session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if uploaded_file and question:
        try:
            # Lecture du fichier en fonction de son type
            if uploaded_file.name.endswith(".txt"):
                document = uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith(".pdf"):
                with pdfplumber.open(uploaded_file) as pdf:
                    document = "\n".join([page.extract_text() or "" for page in pdf.pages])
            else:
                st.error("❌ Format non supporté.")
                st.stop()

            st.markdown("#### 🔎 Extraction facultative des transactions")
            if st.button("Extraire les transactions du relevé"):
                transactions_df = extract_transactions(document)
                if transactions_df is not None:
                    st.markdown("**Transactions détectées :**")
                    st.dataframe(transactions_df)
                else:
                    st.info("Aucune transaction n'a pu être détectée automatiquement.")

            st.markdown("#### 📢 Lancement de l'analyse du relevé bancaire")
            if st.button("Lancer l'analyse du relevé"):
                if mode_test:
                    # Réponse simulée pour le mode test
                    fake_response = (
                        "Réponse simulée : Votre relevé montre des dépenses excessives en abonnements et frais de livraison. "
                        "Il serait judicieux de revoir ces postes pour optimiser votre budget mensuel."
                    )
                    st.success(fake_response)
                    st.session_state.chat_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "user": question,
                        "bot": fake_response
                    })
                    send_telegram_message(f"[TEST IA] {fake_response}")
                else:
                    # Construction du prompt pour l'IA avec instructions détaillées
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
                            if 'choices' in part and len(part['choices']) > 0:
                                content = part['choices'][0].get('delta', {}).get('content', '')
                                full_response += content
                                placeholder.markdown(full_response)

                        # Ajouter la réponse complète à l'historique
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



# ================= TAB 2 : Bot de Trading BTC/USDT =================
with tab2:
    st.title("📈 Bot de Trading BTC/USDT")
    st.markdown(
        """
        **Bienvenue dans le Bot de Trading BTC/USDT !**

        Ce module vous permet d'analyser le marché du Bitcoin en temps réel grâce à des indicateurs techniques avancés.

        **Utilisation :**
        1. **Période d'analyse :** Sélectionnez la durée (en jours) pour analyser l'historique OHLC (1, 7 ou 30 jours).
        2. **Paramètres techniques :**
           - **Seuil RSI :** Ajustez le niveau critique du RSI (par défaut 50).
           - **Décalage MACD :** Définissez un décalage pour ajuster la sensibilité du MACD.
        3. **Lancement de l'analyse :** Cliquez sur le bouton pour récupérer les données, calculer les indicateurs (EMA, RSI, MACD, Bollinger Bands) et afficher des graphiques détaillés.
        4. **Notification :** En cas de signal BUY ou SELL ok, une notification est envoyée via Telegram.
        """
    )

    # Sélection de la période d'analyse
    days = st.selectbox("Sélectionnez la période d'analyse (en jours)", options=[1, 7, 30], index=0)

    st.markdown("#### Personnalisez vos paramètres d'analyse")
    rsi_threshold = st.slider("Seuil RSI (défaut 50)", 30, 70, 50)
    macd_offset = st.number_input("Décalage minimal MACD (défaut 0)", value=0.0, step=0.1)

    @st.cache_data
    def get_btc_price():
        """Récupère le prix actuel du BTC/USDT depuis CoinGecko."""
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
        """Récupère les données OHLC du BTC/USDT sur la période sélectionnée depuis CoinGecko."""
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
        """
        Calcule les indicateurs techniques suivants :
        - EMA50 et EMA200 pour suivre la tendance à court et long terme.
        - RSI pour mesurer le momentum.
        - MACD et Signal MACD (avec décalage) pour détecter les changements de tendance.
        - Bollinger Bands pour évaluer la volatilité.
        """
        df["EMA50"] = df["close"].ewm(span=50).mean()
        df["EMA200"] = df["close"].ewm(span=200).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean() + macd_offset
        # Calcul des Bollinger Bands sur une moyenne mobile de 20 périodes
        df["MA20"] = df["close"].rolling(window=20).mean()
        df["BB_upper"] = df["MA20"] + 2 * df["close"].rolling(window=20).std()
        df["BB_lower"] = df["MA20"] - 2 * df["close"].rolling(window=20).std()
        return df

    def analyse_trading(df):
        """
        Analyse les indicateurs pour générer un signal de trading :
        - BUY : tendance haussière confirmée.
        - SELL : tendance baissière confirmée.
        - NEUTRAL : aucun signal clair.
        """
        last = df.iloc[-1]
        if last["close"] > last["EMA50"] > last["EMA200"] and last["RSI"] > rsi_threshold and last["MACD"] > last["MACD_Signal"]:
            return "BUY", "Signal d'achat détecté : tendance haussière confirmée par les indicateurs."
        elif last["close"] < last["EMA50"] < last["EMA200"] and last["RSI"] < rsi_threshold and last["MACD"] < last["MACD_Signal"]:
            return "SELL", "Signal de vente détecté : tendance baissière confirmée par les indicateurs."
        else:
            return "NEUTRAL", "Aucun signal clair détecté actuellement."

    def plot_trading_chart(df):
        """
        Affiche deux graphiques :
        - Le premier montre le prix du BTC avec EMA50, EMA200 et Bollinger Bands.
        - Le second affiche les oscillateurs RSI et MACD.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # Graphique du prix et indicateurs de tendance
        ax1.plot(df.index, df["close"], label="Prix BTC", color="black")
        ax1.plot(df.index, df["EMA50"], label="EMA 50", color="blue")
        ax1.plot(df.index, df["EMA200"], label="EMA 200", color="red")
        ax1.plot(df.index, df["BB_upper"], label="Bollinger Upper", linestyle="--", color="green")
        ax1.plot(df.index, df["BB_lower"], label="Bollinger Lower", linestyle="--", color="green")
        ax1.fill_between(df.index, df["BB_lower"], df["BB_upper"], color="green", alpha=0.1)
        ax1.set_title("Cours BTC/USDT et indicateurs techniques")
        ax1.legend()
        # Graphique des oscillateurs
        ax2.plot(df.index, df["RSI"], label="RSI", color="purple")
        ax2.plot(df.index, df["MACD"], label="MACD", color="orange")
        ax2.plot(df.index, df["MACD_Signal"], label="Signal MACD", linestyle="--", color="gray")
        ax2.set_title("RSI et MACD")
        ax2.legend()
        st.pyplot(fig)

    # Affichage du prix actuel de BTC/USDT
    current_price = get_btc_price()
    st.metric("💰 Prix actuel BTC/USDT", f"${current_price:.2f}")

    # Lancement de l'analyse de trading
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
            - **EMA 50 / EMA 200** : Moyennes mobiles qui indiquent la tendance à court et long terme.
              - **EMA 50** : Réactive aux changements de prix récents.
              - **EMA 200** : Indicateur de tendance à long terme.
            - **RSI** : Mesure le momentum ; un RSI supérieur à 50 suggère une tendance haussière.
            - **MACD** : Différence entre deux moyennes mobiles exponentielles (12 et 26 périodes).
              - **Signal MACD** : Moyenne mobile de la MACD sur 9 périodes (ajustée par le décalage).
            - **Bollinger Bands** : Basées sur une moyenne mobile sur 20 périodes et ±2 écarts-types pour mesurer la volatilité.
            """
        )

# ================= TAB 3 : Analyse de Portefeuille =================
# ================= TAB 3 : Analyse de Portefeuille =================
# ================= TAB 3 : Analyse de Portefeuille =================
with tab3:
    st.title("📊 Analyse de Portefeuille - Optimisez vos Investissements")
    st.markdown(
        """
        Téléversez votre portefeuille au format CSV pour obtenir une analyse complète :
        - Répartition des actifs.
        - Valeur totale.
        - Indice de diversification.
        - Estimation de la volatilité globale.
        - Analyse des risques.
        - Recommandations d'investissement.
        - Comparaison avec un indice de référence.
        - Simulation de scénarios de marché.
        """
    )
    uploaded_portfolio = st.file_uploader("Téléversez votre portefeuille (CSV: Asset, Quantity, Price)", type=["csv"])

    if uploaded_portfolio is not None:
        try:
            portfolio_df = pd.read_csv(uploaded_portfolio)
            # Vérifiez que les colonnes nécessaires sont présentes
            required_columns = {"Asset", "Quantity", "Price"}
            if not required_columns.issubset(portfolio_df.columns):
                st.error("Le fichier CSV doit contenir les colonnes : Asset, Quantity, Price.")
                st.stop()
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
            st.stop()
    else:
        # Portefeuille exemple si aucun fichier n'est fourni
        data = {'Asset': ['BTC', 'ETH', 'AAPL', 'GOOGL', 'AMZN'],
                'Quantity': [0.5, 2, 10, 5, 8],
                'Price': [60000, 4000, 150, 2800, 3500]}
        portfolio_df = pd.DataFrame(data)
        st.info("Utilisation d'un portefeuille exemple.")

    # Calcul de la valeur par actif et répartition
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

    # Estimation de la volatilité du portefeuille
    volatility = portfolio_df["Value"].std()
    st.write(f"**Volatilité estimée du portefeuille :** {volatility:.2f}")

    # Analyse des risques
    st.subheader("Analyse des Risques")
    st.write("Analyse des risques basée sur la volatilité historique des actifs.")
    # Note : Pour une analyse des risques complète, des données historiques sont nécessaires.

    # Recommandations d'investissement
    st.subheader("Recommandations d'Investissement")
    st.write("Utilisez l'IA pour obtenir des recommandations personnalisées.")
    portfolio_question = st.text_area("Posez votre question sur votre portefeuille :", placeholder="Ex : Comment optimiser ma diversification et réduire le risque ?")
    if portfolio_question:
        portfolio_summary = portfolio_df.to_csv(index=False)
        if mode_test:
            fake_response = (
                "Analyse simulée : Votre portefeuille est globalement équilibré, mais vous pourriez augmenter l'exposition aux actions technologiques "
                "pour améliorer la performance à long terme tout en surveillant le risque global."
            )
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
            try:
                with st.spinner("💬 L'IA analyse votre portefeuille..."):
                    response = client_openai.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        stream=True,
                    )
                    portfolio_response = ""
                    placeholder_portfolio = st.empty()
                    for part in response:
                        if 'choices' in part and len(part['choices']) > 0:
                            content = part['choices'][0].get('delta', {}).get('content', '')
                            portfolio_response += content
                            placeholder_portfolio.markdown(portfolio_response)
                    send_telegram_message(f"📩 Analyse Portefeuille IA :\n{portfolio_response[:400]}...")
            except Exception as e:
                st.error(f"Erreur lors de l'appel à l'API OpenAI : {e}")

    # Comparaison avec un indice de référence
    st.subheader("Comparaison avec un Indice de Référence")
    st.write("Comparaison de la performance du portefeuille avec le S&P 500.")
    # Note : Pour une comparaison complète, des données historiques sont nécessaires.

    # Simulation de scénarios de marché
    st.subheader("Simulation de Scénarios de Marché")
    st.write("Simulez différents scénarios de marché pour évaluer l'impact sur votre portefeuille.")
    # Note : Pour une simulation complète, des données historiques et des modèles de prévision sont nécessaires.

    # Graphique circulaire de répartition des actifs
    st.subheader("Répartition des Actifs")
    fig_port, ax_port = plt.subplots(figsize=(6, 6))
    ax_port.pie(portfolio_df["Value"], labels=portfolio_df["Asset"], autopct='%1.1f%%', startangle=90)
    ax_port.axis('equal')
    st.pyplot(fig_port)

    # Graphique à barres des valeurs par actif
    st.subheader("Valeur par Actif")
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    ax_bar.bar(portfolio_df["Asset"], portfolio_df["Value"], color="skyblue")
    ax_bar.set_title("Valeur par Actif")
    ax_bar.set_ylabel("Valeur ($)")
    st.pyplot(fig_bar)

    # Option de téléchargement du rapport complet
    csv_report = portfolio_df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger le rapport CSV", data=csv_report, file_name="rapport_portefeuille.csv", mime="text/csv")


# ================= FIN DU CODE =================
