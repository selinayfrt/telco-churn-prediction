from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from Telco_Customer_Churn_Feature_Engineering import catboost_model, X_test

# ── LLM ─────────────────────────────────────────────────────────────────────

beyin = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.7
)

# ── ARAÇLAR ─────────────────────────────────────────────────────────────────

class ChurnPredictionTool(BaseTool):
    name: str = "churn_prediction_tool"
    description: str = (
        "Belirli bir müşteri indeksi için detaylı churn tahmini yapar. "
        "Girdi: müşteri indeks numarası (örn. '42'). "
        "Çıktı: tahmin, ayrılma olasılığı ve müşteri profil detayları."
    )

    def _run(self, customer_index: str) -> str:
        try:
            idx = int("".join(filter(str.isdigit, str(customer_index))))
            sample = X_test.iloc[[idx]]
            prob = catboost_model.predict_proba(sample)[0][1]
            pred = catboost_model.predict(sample)[0]
            durum = "Ayrılacak" if pred == 1 else "Ayrılmayacak"
            row = sample.iloc[0]

            detaylar = []
            if "tenure" in row.index:
                detaylar.append(f"Müşteri süresi: {int(row['tenure'])} ay")
            if "MonthlyCharges" in row.index:
                detaylar.append(f"Aylık ücret: ${row['MonthlyCharges']:.2f}")
            if "TotalCharges" in row.index:
                detaylar.append(f"Toplam ödeme: ${row['TotalCharges']:.2f}")
            if "Contract_Month-to-month" in row.index and row["Contract_Month-to-month"] == 1:
                detaylar.append("Sözleşme: Aylık")
            if "Contract_Two year" in row.index and row["Contract_Two year"] == 1:
                detaylar.append("Sözleşme: 2 yıllık")
            if "InternetService_Fiber optic" in row.index and row["InternetService_Fiber optic"] == 1:
                detaylar.append("İnternet: Fiber optik")
            if "PaymentMethod_Electronic check" in row.index and row["PaymentMethod_Electronic check"] == 1:
                detaylar.append("Ödeme: Elektronik çek")
            if "NEW_TotalServices" in row.index:
                detaylar.append(f"Toplam hizmet: {int(row['NEW_TotalServices'])}")

            return (
                f"Müşteri {idx} Profili:\n"
                f"Tahmin: {durum}\n"
                f"Ayrılma Olasılığı: %{round(prob * 100, 2)}\n"
                f"Detaylar: {', '.join(detaylar)}"
            )
        except Exception as e:
            return f"Hata: {str(e)}"


class TopRiskyCustomersTool(BaseTool):
    name: str = "top_risky_customers_tool"
    description: str = (
        "Tüm test setindeki en yüksek churn riskli N müşteriyi bulur ve indekslerini döner. "
        "Girdi: gösterilecek müşteri sayısı (örn. '5'). "
        "Çıktı: indeks ve ayrılma olasılığı listesi."
    )

    def _run(self, n: str = "5") -> str:
        try:
            import pandas as pd
            n = int("".join(filter(str.isdigit, str(n))) or "5")
            probs = catboost_model.predict_proba(X_test)[:, 1]
            risk_df = pd.DataFrame({"index": range(len(probs)), "prob": probs})
            top_n = risk_df.nlargest(n, "prob")
            sonuc = f"En Riskli {n} Müşteri:\n"
            for _, row in top_n.iterrows():
                sonuc += f"  - İndeks {int(row['index'])}: %{round(row['prob'] * 100, 2)} ayrılma riski\n"
            return sonuc
        except Exception as e:
            return f"Hata: {str(e)}"


# ── AJANLAR ─────────────────────────────────────────────────────────────────

analist_ajan = Agent(
    role="Kıdemli Müşteri Analisti",
    goal="Müşterilerin churn riskini analiz et ve detaylı rapor yaz",
    backstory=(
        "Telekom sektöründe müşteri kaybı analizinde uzman bir veri analistisin. "
        "Sayısal verileri yorumlayarak risk faktörlerini net biçimde raporlarsın."
    ),
    tools=[TopRiskyCustomersTool(), ChurnPredictionTool()],
    verbose=True,
    llm=beyin,
    max_iter=7,
)

stratejist_ajan = Agent(
    role="Pazarlama Stratejisti",
    goal="Analistin raporuna göre her riskli müşteri için kişiselleştirilmiş elde tutma teklifleri üret",
    backstory=(
        "Telekom pazarlamasında uzman bir stratejistsin. "
        "Müşteri profillerine göre indirim, kampanya ve sözleşme teklifleri oluşturursun. "
        "Önerilerini somut ve uygulanabilir tutarsın."
    ),
    tools=[],
    verbose=True,
    llm=beyin,
    max_iter=5,
)

# ── GÖREVLER ─────────────────────────────────────────────────────────────────

analiz_gorevi = Task(
    description="""
    Adım 1: top_risky_customers_tool aracını '5' parametresiyle çağır ve en riskli 5 müşterinin indeksini al.
    Adım 2: Bu 5 indeksin HER BİRİ için churn_prediction_tool aracını ayrı ayrı çağır.
    Adım 3: Her müşteri için şunları açıkla:
      - Ayrılma olasılığı ve tahmin sonucu
      - Profil detayları (süre, ücret, sözleşme tipi vb.)
      - Neden risk altında olduğuna dair somut risk faktörleri
    Tüm çıktıları Türkçe yaz.
    """,
    expected_output="""
    5 müşterinin her biri için ayrı bir bölüm:
    - Müşteri indeksi ve ayrılma olasılığı
    - Profil detayları
    - Risk faktörleri açıklaması
    """,
    agent=analist_ajan,
)

strateji_gorevi = Task(
    description="""
    Analistin raporundaki her müşteri için kişiselleştirilmiş elde tutma teklifi üret.
    Müşteri profiline göre aşağıdakilerden uygun olanları seç:
    - İndirim oranı (%10, %20 veya %30) — aylık ücrete göre belirle
    - Sözleşme yükseltme teklifi (aylıktan yıllığa veya 2 yıllığa geçiş)
    - Ücretsiz hizmet ekleme (teknik destek, online güvenlik, güvenlik paketi)
    - Ödeme yöntemi değiştirme teşviki (elektronik çekten otomatik ödemeye)
    Her müşteri için 2-3 somut öneri yaz.
    Raporun sonunda şirkete yönelik 3 genel stratejik öneri ekle.
    Tüm çıktıları Türkçe yaz.
    """,
    expected_output="""
    Her müşteri için kişiselleştirilmiş pazarlama teklifleri (2-3 öneri).
    Raporun sonunda şirkete 3 stratejik öneri.
    """,
    agent=stratejist_ajan,
    context=[analiz_gorevi],
)

# ── ÇALIŞTIRMA ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ekip = Crew(
        agents=[analist_ajan, stratejist_ajan],
        tasks=[analiz_gorevi, strateji_gorevi],
        verbose=True,
    )

    print("\n" + "=" * 50)
    print("### MULTI-AGENT SİSTEM BAŞLATILIYOR ###")
    print("### Analist + Pazarlama Stratejisti   ###")
    print("=" * 50 + "\n")

    sonuc = ekip.kickoff()

    print("\n" + "=" * 50)
    print("### STRATEJİST RAPORU ###")
    print("=" * 50)
    print(sonuc)