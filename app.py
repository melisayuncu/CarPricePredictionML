
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import shap
from sklearn.preprocessing import LabelEncoder

# ─── SAYFA AYARI ───────────────────────────────────────────
st.set_page_config(
    page_title="Araç Fiyat Tahmini",
    page_icon="🚗",
    layout="wide"
)

# ─── GERÇEK DÜNYA LİSTELERİ ────────────────────────────────
ALL_MANUFACTURERS = sorted([
    "acura", "alfa-romeo", "aston-martin", "audi", "bentley",
    "bmw", "buick", "cadillac", "chevrolet", "chrysler",
    "dacia", "dodge", "ferrari", "fiat", "ford", "genesis",
    "gmc", "harley-davidson", "honda", "hyundai", "infiniti",
    "jaguar", "jeep", "kia", "lamborghini", "land rover",
    "lexus", "lincoln", "lotus", "maserati", "mazda",
    "mclaren", "mercedes-benz", "mercury", "mini", "mitsubishi",
    "morgan", "nissan", "opel", "peugeot", "pontiac",
    "porsche", "ram", "renault", "rivian", "rolls-royce",
    "rover", "saturn", "scion", "seat", "skoda",
    "subaru", "tesla", "toyota", "volkswagen", "volvo",
    "other"
])

ALL_CONDITIONS    = ["excellent", "good", "like new", "new", "fair", "salvage", "unknown"]
ALL_CYLINDERS     = ["3 cylinders", "4 cylinders", "5 cylinders", "6 cylinders",
                     "8 cylinders", "10 cylinders", "12 cylinders", "other", "unknown"]
ALL_FUELS         = ["gas", "diesel", "electric", "hybrid", "other"]
ALL_TRANSMISSIONS = ["automatic", "manual", "other"]
ALL_DRIVES        = ["4wd", "fwd", "rwd", "unknown"]
ALL_TYPES         = ["SUV", "bus", "convertible", "coupe", "hatchback", "mini-van",
                     "offroad", "other", "pickup", "sedan", "truck", "van",
                     "wagon", "unknown"]
ALL_COLORS        = ["black", "blue", "brown", "custom", "green", "grey",
                     "orange", "purple", "red", "silver", "white", "yellow", "unknown"]
ALL_STATES        = ["ak","al","ar","az","ca","co","ct","dc","de","fl",
                     "ga","hi","ia","id","il","in","ks","ky","la","ma",
                     "md","me","mi","mn","mo","ms","mt","nc","nd","ne",
                     "nh","nj","nm","nv","ny","oh","ok","or","pa","ri",
                     "sc","sd","tn","tx","ut","va","vt","wa","wi","wv","wy"]

# ─── MODEL YÜKLEME ─────────────────────────────────────────
@st.cache_resource
def load_models():
    base = "models"
    with open(f"{base}/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(f"{base}/xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(f"{base}/lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open(f"{base}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{base}/cat_unique.json", "r") as f:
        cat_unique = json.load(f)
    with open(f"{base}/shap_values.pkl", "rb") as f:
        shap_values = pickle.load(f)
    with open(f"{base}/X_sample.pkl", "rb") as f:
        X_sample = pickle.load(f)
    return rf, xgb, lr, scaler, cat_unique, shap_values, X_sample

rf_model, xgb_model, lr_model, scaler, cat_unique, shap_values, X_sample = load_models()

# ─── ENCODING YARDIMCI FONKSİYON ───────────────────────────
def safe_encode(col, val, cat_unique):
    known_vals = cat_unique[col]
    if val not in known_vals:
        val = "unknown" if "unknown" in known_vals else known_vals[0]
    le = LabelEncoder()
    le.fit(known_vals)
    return le.transform([val])[0], val

# ─── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("🚗 Araç Fiyat Tahmini")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Sayfa Seç",
    ["Ana Sayfa", "Veri Keşfi", "Model Karşılaştırması", "SHAP Analizi", "Fiyat Tahmini"]
)

# ══════════════════════════════════════════════════════════════
# SAYFA 1 — ANA SAYFA
# ══════════════════════════════════════════════════════════════
if page == "Ana Sayfa":
    st.title("🚗 Craigslist Araç Fiyat Tahmin Uygulaması")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Veri", "426,880")
    col2.metric("Temizleme Sonrası", "367,218")
    col3.metric("Feature Sayısı", "14")
    col4.metric("En İyi R²", "0.8289")

    st.markdown("---")
    st.subheader("📋 Proje Hakkında")
    st.markdown("""
    Bu uygulama, ABD\'nin en büyük ikinci el araç platformu **Craigslist**\'ten
    derlenen veri seti üzerinde makine öğrenmesi modelleri eğiterek araç fiyatı
    tahmini yapmaktadır.
    """)

    st.subheader("🔄 Pipeline")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        1. ✅ Veri Yükleme (426,880 satır, 26 kolon)
        2. ✅ EDA (dağılımlar, korelasyon analizi)
        3. ✅ Veri Temizleme (367,218 satır, 0 eksik)
        4. ✅ Feature Engineering (4 yeni feature)
        """)
    with col2:
        st.markdown("""
        5. ✅ Feature Selection (15 kolon)
        6. ✅ Train/Test Split (80/20)
        7. ✅ Normalizasyon (StandardScaler)
        8. ✅ Model Kurma (LR, RF, XGBoost)
        9. ✅ Model Değerlendirme
        10. ✅ SHAP Analizi
        """)

    st.subheader("📊 Model Sonuçları")
    results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "XGBoost v2"],
        "MAE ($)": [5646, 3439, 3803],
        "RMSE ($)": [8610, 6078, 6437],
        "R²": [0.6567, 0.8289, 0.8081]
    })
    st.dataframe(results, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# SAYFA 2 — VERİ KEŞFİ
# ══════════════════════════════════════════════════════════════
elif page == "Veri Keşfi":
    st.title("🔍 Veri Keşfi")
    st.markdown("---")

    st.subheader("Outlier Temizleme: Korelasyon Karşılaştırması")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    corr_before = pd.DataFrame({
        "price":    [1.00, -0.00,  0.01],
        "year":     [-0.00, 1.00, -0.16],
        "odometer": [0.01, -0.16,  1.00]
    }, index=["price", "year", "odometer"])

    corr_after = pd.DataFrame({
        "price":    [1.00,  0.57, -0.54],
        "year":     [0.57,  1.00, -0.64],
        "odometer": [-0.54, -0.64, 1.00]
    }, index=["price", "year", "odometer"])

    sns.heatmap(corr_before, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title("HAM VERİ Korelasyonu", fontweight="bold")

    sns.heatmap(corr_after, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title("TEMİZ VERİ Korelasyonu", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("""
    Ham veride price-year korelasyonu **-0.00** iken temiz veride **+0.57**\'e yükseldi.
    Bu, outlier\'ların gerçek istatistiksel ilişkileri tamamen maskelediğinin kanıtıdır.
    """)

    st.subheader("Veri Temizleme Özeti")
    col1, col2, col3 = st.columns(3)
    col1.metric("Başlangıç Satır", "426,880")
    col2.metric("Temizleme Sonrası", "367,218")
    col3.metric("Kaybedilen", "59,662 (%14)")

    st.subheader("Eksik Değer Analizi")
    missing_data = pd.DataFrame({
        "Kolon": ["cylinders", "condition", "drive", "paint_color", "type",
                  "manufacturer", "title_status", "model", "fuel", "transmission"],
        "Eksik %": [41.2, 38.0, 30.2, 28.7, 20.9, 3.2, 1.8, 0.9, 0.6, 0.4],
        "Strateji": ["unknown", "unknown", "unknown", "unknown", "unknown",
                     "mod", "mod", "mod", "mod", "mod"]
    })
    st.dataframe(missing_data, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# SAYFA 3 — MODEL KARŞILAŞTIRMASI
# ══════════════════════════════════════════════════════════════
elif page == "Model Karşılaştırması":
    st.title("📊 Model Karşılaştırması")
    st.markdown("---")

    models    = ["Linear Regression", "Random Forest", "XGBoost v2"]
    mae_vals  = [5646, 3439, 3803]
    rmse_vals = [8610, 6078, 6437]
    r2_vals   = [0.6567, 0.8289, 0.8081]
    colors    = ["#e74c3c", "#2ecc71", "#3498db"]

    st.subheader("Metrik Karşılaştırması")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(models, mae_vals, color=colors)
    axes[0].set_title("MAE (Düşük = İyi)", fontweight="bold")
    axes[0].set_ylabel("Dolar ($)")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(mae_vals):
        axes[0].text(i, v+50, f"${v:,}", ha="center", fontweight="bold")

    axes[1].bar(models, rmse_vals, color=colors)
    axes[1].set_title("RMSE (Düşük = İyi)", fontweight="bold")
    axes[1].set_ylabel("Dolar ($)")
    axes[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(rmse_vals):
        axes[1].text(i, v+50, f"${v:,}", ha="center", fontweight="bold")

    axes[2].bar(models, r2_vals, color=colors)
    axes[2].set_title("R² Skoru (Yüksek = İyi)", fontweight="bold")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=15)
    for i, v in enumerate(r2_vals):
        axes[2].text(i, v+0.01, f"{v:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.success("""
    🏆 **Random Forest** tüm metriklerde en iyi performansı gösterdi.
    MAE: $3,439 | RMSE: $6,078 | R²: 0.8289
    Fiyat varyansının %82.9\'unu açıklıyor!
    """)

    st.subheader("Hata Dağılımı Karşılaştırması")
    st.markdown("""
    | Model | Ort. Hata | Std | %95 Aralık |
    |---|---|---|---|
    | Linear Regression | -\$29 | \$8,610 | -\$13,929 / +\$18,106 |
    | Random Forest | -\$29 | \$6,078 | -\$10,442 / +\$11,916 |
    | XGBoost v2 | -\$11 | \$6,437 | -\$10,624 / +\$13,140 |
    """)

# ══════════════════════════════════════════════════════════════
# SAYFA 4 — SHAP ANALİZİ
# ══════════════════════════════════════════════════════════════
elif page == "SHAP Analizi":
    st.title("🔬 SHAP Analizi")
    st.markdown("---")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) analizi, modelin her tahmini için
    hangi feature\'ın ne kadar ve hangi yönde etkili olduğunu gösterir.
    """)

    st.subheader("Ortalama Mutlak SHAP Değerleri")
    shap_df = pd.DataFrame({
        "Feature": ["vehicle_age","odometer","drive","cylinders","fuel",
                    "is_luxury","type","manufacturer","condition",
                    "transmission","age_odometer_ratio","state",
                    "paint_color","is_clean_title"],
        "SHAP ($)": [6798,2855,2203,2198,1912,
                     747,648,577,374,
                     276,220,181,177,109]
    }).sort_values("SHAP ($)", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(shap_df["Feature"], shap_df["SHAP ($)"], color="#2ecc71")
    ax.set_xlabel("Ortalama Mutlak SHAP Değeri ($)")
    ax.set_title("SHAP Feature Importance", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("SHAP Beeswarm Plot")
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)
    shap.summary_plot(shap_values, X_sample, show=False)
    ax.set_title("SHAP Summary — Yön ve Büyüklük", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("SHAP Dependence Plot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    shap.dependence_plot("vehicle_age", shap_values, X_sample,
                         ax=axes[0], show=False)
    axes[0].set_title("vehicle_age → Fiyata Etkisi", fontweight="bold")
    shap.dependence_plot("odometer", shap_values, X_sample,
                         ax=axes[1], show=False)
    axes[1].set_title("odometer → Fiyata Etkisi", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("""
    **Bulgular:**
    - vehicle_age tek başına fiyatı ortalama **$6,798** etkiliyor
    - Genç araçlar fiyatı **+$25,000**\'e kadar artırabilirken yaşlı araçlar **-$10,000**\'e kadar düşürebiliyor
    - odometer arttıkça fiyat üzerindeki etkisi güçlü negatif yönde seyrediyor
    """)

# ══════════════════════════════════════════════════════════════
# SAYFA 5 — FİYAT TAHMİNİ
# ══════════════════════════════════════════════════════════════
elif page == "Fiyat Tahmini":
    st.title("💰 Araç Fiyat Tahmini")
    st.markdown("---")
    st.markdown("Araç özelliklerini girerek **Random Forest** modeliyle fiyat tahmini yapın.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Marka**")
        manufacturer_select = st.selectbox(
            "Listeden seç",
            ["Listeden seç..."] + ALL_MANUFACTURERS,
            key="mfr_select"
        )
        manufacturer_custom = st.text_input(
            "Listede yok mu? Buraya yaz",
            placeholder="örn: seat, dacia, skoda...",
            key="mfr_custom"
        )
        # Öncelik: serbest metin > dropdown
        if manufacturer_custom.strip():
            manufacturer = manufacturer_custom.strip().lower()
        elif manufacturer_select != "Listeden seç...":
            manufacturer = manufacturer_select
        else:
            manufacturer = "other"

        condition    = st.selectbox("Durum", ALL_CONDITIONS)
        cylinders    = st.selectbox("Silindir", ALL_CYLINDERS)
        fuel         = st.selectbox("Yakıt Tipi", ALL_FUELS)
        transmission = st.selectbox("Şanzıman", ALL_TRANSMISSIONS)

    with col2:
        drive       = st.selectbox("Çekiş", ALL_DRIVES)
        type_       = st.selectbox("Araç Tipi", ALL_TYPES)
        paint_color = st.selectbox("Renk", ALL_COLORS)
        state       = st.selectbox("Eyalet", ALL_STATES)

    with col3:
        vehicle_age = st.slider("Araç Yaşı (yıl)", 2, 34, 10)
        odometer    = st.number_input("Kilometre (mil)", 0, 300000, 50000, step=1000)

    st.markdown("---")

    if st.button("🔍 Fiyat Tahmin Et", use_container_width=True):

        luxury_brands = ["bmw", "mercedes-benz", "audi", "lexus",
                         "porsche", "jaguar", "land rover", "cadillac"]
        is_luxury          = 1 if manufacturer in luxury_brands else 0
        is_clean_title     = 1
        age_odometer_ratio = odometer / (vehicle_age + 1)

        cat_cols_order = ["manufacturer", "condition", "cylinders", "fuel",
                          "transmission", "drive", "type", "paint_color", "state"]
        cat_vals_order = [manufacturer, condition, cylinders, fuel,
                          transmission, drive, type_, paint_color, state]

        encoded  = {}
        warnings = []

        for col, val in zip(cat_cols_order, cat_vals_order):
            enc_val, used_val = safe_encode(col, val, cat_unique)
            encoded[col] = enc_val
            if used_val != val:
                warnings.append(f"**{col}**: \"{val}\" → \"{used_val}\" olarak eşleştirildi")

        if warnings:
            st.warning(
                "⚠️ Bazı değerler eğitim verisinde bulunmadığından eşleştirildi:\n\n" +
                "\n".join(warnings)
            )

        num_raw    = np.array([[odometer, vehicle_age, age_odometer_ratio]])
        num_scaled = scaler.transform(num_raw)

        input_df = pd.DataFrame([{
            "manufacturer":       encoded["manufacturer"],
            "condition":          encoded["condition"],
            "cylinders":          encoded["cylinders"],
            "fuel":               encoded["fuel"],
            "odometer":           num_scaled[0][0],
            "transmission":       encoded["transmission"],
            "drive":              encoded["drive"],
            "type":               encoded["type"],
            "paint_color":        encoded["paint_color"],
            "state":              encoded["state"],
            "vehicle_age":        num_scaled[0][1],
            "age_odometer_ratio": num_scaled[0][2],
            "is_luxury":          is_luxury,
            "is_clean_title":     is_clean_title
        }])

        prediction = rf_model.predict(input_df)[0]
        prediction = max(500, prediction)

        st.success(f"### 🚗 Tahmini Fiyat: ${prediction:,.0f}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Tahmini Fiyat", f"${prediction:,.0f}")
        col2.metric("Model", "Random Forest")
        col3.metric("R² Skoru", "0.8289")

        st.markdown(f"""
        **Girilen Özellikler:**
        - Marka: {manufacturer} | Durum: {condition} | Yakıt: {fuel}
        - Araç Yaşı: {vehicle_age} yıl | Kilometre: {odometer:,} mil
        - Lüks Araç: {"Evet ✅" if is_luxury else "Hayır"}
        """)
