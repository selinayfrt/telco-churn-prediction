import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os

# ── Sayfa Ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analiz Sistemi",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Koyu Tema CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:       #0a0a0f;
    --surface:  #13131a;
    --border:   #1e1e2e;
    --accent:   #7c3aed;
    --accent2:  #06b6d4;
    --danger:   #ef4444;
    --safe:     #22c55e;
    --warn:     #f59e0b;
    --text:     #e2e8f0;
    --muted:    #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 15px;
    padding: 12px 28px;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}
.risk-high   { border-left: 4px solid var(--danger) !important; }
.risk-medium { border-left: 4px solid var(--warn)   !important; }
.risk-low    { border-left: 4px solid var(--safe)   !important; }

.tag {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.4);
    color: #a78bfa;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}

.rapor-kutu {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent2);
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
    line-height: 1.8;
    color: var(--text);
    font-size: 15px;
}

h1,h2,h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }

[data-testid="stDataFrame"] { background: var(--surface) !important; }

div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Model Yükle ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from Telco_Customer_Churn_Feature_Engineering import catboost_model, X_test
    return catboost_model, X_test


# load_llm kaldırıldı - model string direkt kullanılıyor


try:
    model, X_test = load_model()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    st.error(f"Model yüklenemedi: {e}")


# ── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────
def risk_rengi(prob):
    if prob >= 0.7:   return "#ef4444", "🔴 YÜKSEK RİSK", "risk-high"
    elif prob >= 0.4: return "#f59e0b", "🟡 ORTA RİSK",   "risk-medium"
    else:             return "#22c55e", "🟢 DÜŞÜK RİSK",  "risk-low"


def risk_faktorleri(row):
    f = []
    if "tenure" in row.index and row["tenure"] < 12:
        f.append(f"Düşük müşteri süresi ({int(row['tenure'])} ay)")
    if "MonthlyCharges" in row.index and row["MonthlyCharges"] > 70:
        f.append(f"Yüksek aylık ücret (${row['MonthlyCharges']:.2f})")
    if "Contract_Month-to-month" in row.index and row["Contract_Month-to-month"] == 1:
        f.append("Aylık sözleşme (bağlılık yok)")
    if "InternetService_Fiber optic" in row.index and row["InternetService_Fiber optic"] == 1:
        f.append("Fiber optik internet (yüksek churn segmenti)")
    if "OnlineSecurity_No" in row.index and row["OnlineSecurity_No"] == 1:
        f.append("Online güvenlik hizmeti yok")
    if "PaymentMethod_Electronic check" in row.index and row["PaymentMethod_Electronic check"] == 1:
        f.append("Elektronik çek ödeme (riskli yöntem)")
    return f if f else ["Belirgin risk faktörü yok"]


def gauge_chart(prob, title="Risk Skoru"):
    renk, _, _ = risk_rengi(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        title={"text": title, "font": {"color": "#e2e8f0", "family": "Syne", "size": 16}},
        number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b"},
            "bar": {"color": renk},
            "bgcolor": "#13131a",
            "bordercolor": "#1e1e2e",
            "steps": [
                {"range": [0,  40], "color": "rgba(34,197,94,0.15)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                {"range": [70,100], "color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {"line": {"color": renk, "width": 3}, "value": prob * 100}
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font_color="#e2e8f0", height=280, margin=dict(t=40, b=10)
    )
    return fig

def shap_grafik(musteri_id):
    import shap
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    explainer = shap.TreeExplainer(model)
    row = X_test.iloc[[musteri_id]]
    shap_values = explainer.shap_values(row)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#13131a")

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=row.iloc[0].values,
            feature_names=X_test.columns.tolist()
        ),
        max_display=10,
        show=False
    )

    plt.tight_layout()
    return fig

def ai_rapor_olustur(musteri_id, prob, faktörler):
    """CrewAI ajanını çalıştırır, Türkçe rapor döner."""
    from crewai import Agent, Task, Crew
    from crewai.tools import BaseTool

    class ChurnTool(BaseTool):
        name: str = "churn_prediction_tool"
        description: str = "Müşteri indeksiyle churn tahmini yapar."

        def _run(self, customer_index: str) -> str:
            idx = int("".join(filter(str.isdigit, str(customer_index))))
            s = X_test.iloc[[idx]]
            p = model.predict_proba(s)[0][1]
            pred = model.predict(s)[0]
            durum = "Ayrılacak" if pred == 1 else "Ayrılmayacak"
            row = s.iloc[0]
            facts = risk_faktorleri(row)

            # Müşteriye özel detaylar
            detaylar = []
            if "tenure" in row.index:
                detaylar.append(f"Müşteri süresi: {int(row['tenure'])} ay")
            if "MonthlyCharges" in row.index:
                detaylar.append(f"Aylık ücret: ${row['MonthlyCharges']:.2f}")
            if "TotalCharges" in row.index:
                detaylar.append(f"Toplam ödeme: ${row['TotalCharges']:.2f}")
            if "Contract_Month-to-month" in row.index and row["Contract_Month-to-month"] == 1:
                detaylar.append("Sözleşme: Aylık (kısa vadeli)")
            if "Contract_Two year" in row.index and row["Contract_Two year"] == 1:
                detaylar.append("Sözleşme: 2 yıllık (uzun vadeli)")
            if "InternetService_Fiber optic" in row.index and row["InternetService_Fiber optic"] == 1:
                detaylar.append("İnternet: Fiber optik")
            if "InternetService_DSL" in row.index and row["InternetService_DSL"] == 1:
                detaylar.append("İnternet: DSL")
            if "NEW_TotalServices" in row.index:
                detaylar.append(f"Kullandığı hizmet sayısı: {int(row['NEW_TotalServices'])}")

            return (
                f"Müşteri {idx} Detaylı Profil:\n"
                f"Tahmin: {durum}\n"
                f"Ayrılma Olasılığı: %{round(p * 100, 2)}\n"
                f"Müşteri Bilgileri: {', '.join(detaylar)}\n"
                f"Risk Faktörleri: {', '.join(facts)}"
            )

    MODEL_ADI = "groq/llama-3.3-70b-versatile"

    ajan = Agent(
        role="Kıdemli Müşteri Analisti",
        goal="Churn riskini analiz et ve profesyonel Türkçe rapor yaz",
        backstory="Telekom sektöründe müşteri kaybı analizinde uzmanlaşmış bir veri analistisin.",
        tools=[ChurnTool()],
        verbose=False,
        llm=MODEL_ADI,
        max_iter=3,
    )

    gorev = Task(
        description=(
            f"{musteri_id} numaralı müşteriyi analiz et. "
            f"Risk faktörlerini açıkla ve müşteriyi elde tutmak için "
            f"şirkete somut öneriler sun. Türkçe yaz."
        ),
        expected_output="Kısa ve profesyonel Türkçe analiz raporu.",
        agent=ajan,
    )

    ekip = Crew(agents=[ajan], tasks=[gorev])
    return str(ekip.kickoff())

def multi_agent_rapor_olustur(musteri_id, prob, faktörler):
    """Analist + Stratejist ajanı çalıştırır."""
    from crewai import Agent, Task, Crew
    from crewai.tools import BaseTool

    class ChurnTool(BaseTool):
        name: str = "churn_prediction_tool"
        description: str = "Müşteri indeksiyle detaylı churn tahmini yapar."
        def _run(self, customer_index: str) -> str:
            idx = int("".join(filter(str.isdigit, str(customer_index))))
            s = X_test.iloc[[idx]]
            p = model.predict_proba(s)[0][1]
            pred = model.predict(s)[0]
            durum = "Ayrılacak" if pred == 1 else "Ayrılmayacak"
            row = s.iloc[0]
            detaylar = []
            if "tenure" in row.index:
                detaylar.append(f"Müşteri süresi: {int(row['tenure'])} ay")
            if "MonthlyCharges" in row.index:
                detaylar.append(f"Aylık ücret: ${row['MonthlyCharges']:.2f}")
            if "TotalCharges" in row.index:
                detaylar.append(f"Toplam ödeme: ${row['TotalCharges']:.2f}")
            if "Contract_Month-to-month" in row.index and row["Contract_Month-to-month"] == 1:
                detaylar.append("Sözleşme: Aylık")
            if "InternetService_Fiber optic" in row.index and row["InternetService_Fiber optic"] == 1:
                detaylar.append("İnternet: Fiber optik")
            if "PaymentMethod_Electronic check" in row.index and row["PaymentMethod_Electronic check"] == 1:
                detaylar.append("Ödeme: Elektronik çek")
            if "NEW_TotalServices" in row.index:
                detaylar.append(f"Toplam hizmet: {int(row['NEW_TotalServices'])}")
            return (
                f"Müşteri {idx} Profili:\n"
                f"Tahmin: {durum}\n"
                f"Ayrılma Olasılığı: %{round(p*100,2)}\n"
                f"Detaylar: {', '.join(detaylar)}"
            )

    MODEL_ADI = "groq/llama-3.3-70b-versatile"

    analist = Agent(
        role="Kıdemli Müşteri Analisti",
        goal="Müşterinin churn riskini analiz et",
        backstory="Telekom sektöründe uzman veri analistisin.",
        tools=[ChurnTool()],
        verbose=False,
        llm=MODEL_ADI,
        max_iter=3,
    )

    stratejist = Agent(
        role="Pazarlama Stratejisti",
        goal="Riskli müşteri için kişiselleştirilmiş teklif üret",
        backstory="Telekom pazarlamasında uzman stratejistsin. Somut ve uygulanabilir teklifler üretirsin.",
        tools=[],
        verbose=False,
        llm=MODEL_ADI,
        max_iter=3,
    )

    analiz_gorevi = Task(
        description=f"{musteri_id} numaralı müşteriyi analiz et. Risk faktörlerini açıkla. Türkçe yaz.",
        expected_output="Müşteri profili ve risk faktörleri raporu.",
        agent=analist,
    )

    strateji_gorevi = Task(
        description=f"""
        Analistin raporuna göre müşteri {musteri_id} için kişiselleştirilmiş teklif üret.
        Müşteri profiline göre şunlardan uygun olanları öner:
        - İndirim oranı (%10, %20, %30)
        - Sözleşme yükseltme teklifi
        - Ücretsiz hizmet ekleme
        - Ödeme yöntemi değiştirme teşviki
        2-3 somut öneri yaz. Türkçe yaz.
        """,
        expected_output="Kişiselleştirilmiş pazarlama teklifleri.",
        agent=stratejist,
        context=[analiz_gorevi],
    )

    ekip = Crew(
        agents=[analist, stratejist],
        tasks=[analiz_gorevi, strateji_gorevi],
        verbose=False,
    )
    return str(ekip.kickoff())
# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Churn Analiz")
    st.markdown("---")
    sayfa = st.radio(
        "Sayfa Seç",
        ["🔍 Tek Müşteri Analizi", "📊 Risk Dashboard"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    if MODEL_OK:
        st.success(f"✅ Model aktif\n\n`{X_test.shape[0]}` müşteri verisi")
    st.markdown("---")
    st.markdown(
        "<span style='color:#64748b;font-size:12px'>CatBoost + CrewAI + Streamlit</span>",
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════════════
# SAYFA 1 — TEK MÜŞTERİ ANALİZİ
# ════════════════════════════════════════════════════════════════════════════
if sayfa == "🔍 Tek Müşteri Analizi":
    st.title("Müşteri Churn Analizi")
    st.markdown("<p style='color:#64748b'>Müşteri ID'si girerek anlık risk tahmini alın.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    col_input, col_empty = st.columns([1, 2])
    with col_input:
        musteri_id = st.number_input(
            "Müşteri İndeksi",
            min_value=0,
            max_value=len(X_test) - 1 if MODEL_OK else 9999,
            value=10,
            step=1
        )
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        with col_btn1:
            analiz_btn = st.button("🔍 Analiz Et")
        with col_btn2:
            rapor_btn = st.button("🤖 AI Raporu")
        with col_btn3:
            strateji_btn = st.button("🎯 Strateji")
        with col_btn4:
            shap_btn = st.button("🧠 SHAP")

    if (analiz_btn or rapor_btn or strateji_btn or shap_btn) and MODEL_OK:
        row = X_test.iloc[[musteri_id]]
        prob = model.predict_proba(row)[0][1]
        pred = model.predict(row)[0]
        renk, etiket, klass = risk_rengi(prob)
        faktörler = risk_faktorleri(row.iloc[0])

        st.markdown("---")

        # Gauge + Özet
        col_gauge, col_ozet = st.columns([1, 1])
        with col_gauge:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)
        with col_ozet:
            st.markdown(f"""
            <div class="metric-card {klass}">
                <h3 style="margin:0 0 8px 0">Müşteri #{musteri_id}</h3>
                <p style="font-size:24px;font-weight:800;color:{renk};margin:0">{etiket}</p>
                <p style="color:#64748b;margin:8px 0 16px 0">
                    Ayrılma Olasılığı: <b style="color:{renk}">{round(prob*100,2)}%</b>
                </p>
                <p style="color:#94a3b8;margin:0 0 8px 0;font-size:13px">⚠️ Risk Faktörleri:</p>
                {"".join(f'<span class="tag">{f}</span>' for f in faktörler)}
            </div>
            """, unsafe_allow_html=True)

        # Müşteri Bilgileri Tablosu
        st.markdown("#### 📋 Müşteri Özellikleri")
        df_goster = row.T.reset_index()
        df_goster.columns = ["Özellik", "Değer"]
        df_goster = df_goster[df_goster["Değer"] != 0]
        st.dataframe(df_goster, use_container_width=True, hide_index=True)

        # AI Raporu
        if rapor_btn:
            st.markdown("---")
            st.markdown("#### 🤖 AI Analiz Raporu")
            with st.spinner("Ajan rapor hazırlıyor, lütfen bekleyin..."):
                try:
                    rapor = ai_rapor_olustur(musteri_id, prob, faktörler)
                    st.markdown(
                        f'<div class="rapor-kutu">{rapor}</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Rapor oluşturulamadı: {e}")

        # Strateji Raporu
        if strateji_btn:
            st.markdown("---")
            st.markdown("#### 🎯 Analist + Stratejist Raporu")
            with st.spinner("İki ajan çalışıyor, lütfen bekleyin..."):
                try:
                    rapor = multi_agent_rapor_olustur(musteri_id, prob, faktörler)
                    st.markdown(
                        f'<div class="rapor-kutu">{rapor}</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Rapor oluşturulamadı: {e}")

        if shap_btn:
            st.markdown("---")
            st.markdown("#### 🧠 SHAP — Model Açıklanabilirliği")
            st.markdown("<p style='color:#64748b'>Modelin bu müşteri için neden bu tahmini yaptığını gösterir.</p>",
                        unsafe_allow_html=True)
            with st.spinner("SHAP değerleri hesaplanıyor..."):
                try:
                    fig = shap_grafik(musteri_id)
                    st.pyplot(fig)
                    st.markdown("""
                    <div class="rapor-kutu">
                    <b>SHAP Grafiği Nasıl Okunur?</b><br>
                    🔴 Kırmızı çubuklar → Ayrılma riskini <b>artıran</b> faktörler<br>
                    🔵 Mavi çubuklar → Ayrılma riskini <b>azaltan</b> faktörler<br>
                    Çubuk ne kadar uzunsa, o faktörün etkisi o kadar büyük.
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"SHAP hesaplanamadı: {e}")


# ════════════════════════════════════════════════════════════════════════════
# SAYFA 2 — RİSK DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
else:
    st.title("Risk Dashboard")
    st.markdown("<p style='color:#64748b'>Sistemdeki en yüksek churn riskli müşteriler.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    if MODEL_OK:
        n_goster = st.slider("Kaç müşteri gösterilsin?", 3, 20, 5)

        with st.spinner("Analiz yapılıyor..."):
            tum_probs = model.predict_proba(X_test)[:, 1]
            risk_df = pd.DataFrame({
                "Müşteri İndeks": range(len(tum_probs)),
                "Ayrılma Riski (%)": (tum_probs * 100).round(2)
            }).nlargest(n_goster, "Ayrılma Riski (%)")

        # Genel Metrikler
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 Yüksek Riskli", f"{(tum_probs >= 0.7).sum()} müşteri")
        with col2:
            st.metric("🟡 Orta Riskli",   f"{((tum_probs >= 0.4) & (tum_probs < 0.7)).sum()} müşteri")
        with col3:
            st.metric("🟢 Düşük Riskli",  f"{(tum_probs < 0.4).sum()} müşteri")

        st.markdown("---")
        st.markdown(f"#### 🏆 En Riskli {n_goster} Müşteri")

        cols = st.columns(min(n_goster, 3))
        for i, (_, satir) in enumerate(risk_df.iterrows()):
            idx  = int(satir["Müşteri İndeks"])
            prob = satir["Ayrılma Riski (%)"] / 100
            with cols[i % 3]:
                st.plotly_chart(gauge_chart(prob, f"Müşteri #{idx}"),
                                use_container_width=True)
                faktörler = risk_faktorleri(X_test.iloc[idx])
                st.markdown(
                    "".join(f'<span class="tag">{f}</span>' for f in faktörler[:2]),
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("#### 📊 Tüm Risk Listesi")
        st.dataframe(risk_df, use_container_width=True, hide_index=True)