# Laporan Proyek Machine Learning - Estimasi Harga Mobil Menggunakan Machine Learning

## Domain Proyek
Perkembangan teknologi dan digitalisasi telah membawa perubahan signifikan pada
berbagai aspek kehidupan, termasuk sektor otomotif. Salah satu perubahan yang
paling menonjol adalah meningkatnya aktivitas jual beli mobil, baik mobil baru
maupun bekas, secara online. Permintaan terhadap mobil bekas terus meningkat
setiap tahunnya, seiring dengan pertumbuhan ekonomi dan meningkatnya keperluan
masyarakat.

Namun, tantangan terbesar dalam transaksi jual beli mobil bekas adalah
menentukan harga yang wajar dan sesuai dengan kondisi kendaraan. Banyak faktor
yang memengaruhi harga sebuah mobil, seperti merek, model, tahun pembuatan,
kapasitas mesin, jenis bahan bakar, transmisi, serta popularitas mobil tersebut
di pasar. Penjual dan pembeli seringkali menghadapi ketidakpastian dalam
menetapkan atau menilai harga pasar yang sesuai.

Dengan memanfaatkan data historis penjualan mobil beserta fitur-fiturnya,
machine learning dapat digunakan untuk membangun model prediksi harga mobil
secara otomatis dan objektif. Model ini membantu calon pembeli dan penjual dalam
menetapkan harga jual yang kompetitif berdasarkan tren pasar.

## Business Understanding

### Problem Statements
1. Sulitnya menentukan harga mobil bekas yang wajar dan sesuai dengan kondisi
kendaraan.
2. Banyaknya faktor teknis dan non-teknis yang memengaruhi harga mobil yang
sulit dianalisis secara manual.

### Goals
1. Mengembangkan model machine learning yang mampu memprediksi harga mobil bekas
secara akurat berdasarkan data historis dan spesifikasi kendaraan.
2. Menyediakan alat bantu analisis harga mobil dengan mempertimbangkan berbagai
fitur yang memengaruhi harga untuk mengurangi bias subjektif dalam penilaian
harga.

### Solution Statements
- Menggunakan tiga algoritma machine learning: Random Forest, Gradient Boosting,
dan MLP Regressor untuk membandingkan performa dalam memprediksi harga mobil.
- Melakukan preprocessing data seperti pengisian missing value, normalisasi data
numerik dengan StandardScaler, dan encoding data kategorikal dengan
OneHotEncoder.
- Melakukan evaluasi performa model menggunakan RMSE, MAE, dan R2 Score.

## Data Understanding

### Visualisasi Distribusi MSRP
Visualisasi histogram berikut digunakan untuk melihat distribusi nilai MSRP. 
Histogram akan menunjukkan seberapa tersebar harga mobil dalam dataset. 
Kurva KDE (Kernel Density Estimation) ditambahkan untuk memperlihatkan bentuk 
distribusi data secara halus.

```python
plt.figure(figsize=(8,5))
sns.histplot(df['MSRP'], kde=True, bins=40)
plt.title('Distribusi Harga Mobil (MSRP)')
plt.xlabel('Harga (MSRP)')
plt.ylabel('Frekuensi')
plt.show()
```

### Heatmap Korelasi Numerik
Visualisasi ini digunakan untuk memahami korelasi antar fitur numerik, 
yang membantu mengidentifikasi hubungan linier antara variabel. 
Korelasi tinggi (positif atau negatif) dapat berdampak pada kinerja model tertentu.

```python
df_num = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10,7))
sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap='Blues')
plt.title('Korelasi Fitur Numerik')
plt.show()
```

Dataset yang digunakan terdiri dari 11.914 baris dan 16 kolom, diperoleh dari Kaggle:
[sumber](https://www.kaggle.com/datasets/rupindersinghrana/car-features-and-
prices-dataset)

### Deskripsi Fitur:
-Make: Merek mobil (contoh: BMW, Toyota, Ford)
-Model: Model mobil (contoh: Camry, Civic, Accord)
-Year: Tahun pembuatan mobil
-Engine Fuel Type: Jenis bahan bakar mesin (contoh: gasoline, diesel)
-Engine HP: Tenaga mesin dalam satuan horsepower
-Engine Cylinders: Jumlah silinder pada mesin
-Transmission Type: Jenis transmisi mobil (contoh: automatic, manual)
-Driven_Wheels: Jenis penggerak roda (contoh: front wheel drive)
-Number of Doors: Jumlah pintu kendaraan
-Market Category: Kategori pasar mobil (contoh: Luxury, Crossover)
-Vehicle Size: Ukuran kendaraan (Compact, Midsize, Large)
-Vehicle Style: Tipe bodi kendaraan (Sedan, SUV, Convertible)
-highway MPG: Konsumsi bahan bakar di jalan tol (miles per gallon)
-city MPG: Konsumsi bahan bakar di dalam kota
-Popularity: Indeks popularitas mobil berdasarkan data pasar
-MSRP: Manufacturer's Suggested Retail Price, target variabel berupa harga mobil yang disarankan oleh produsen

Exploratory data analysis menunjukkan bahwa MSRP memiliki distribusi right-skewed dan beberapa outlier. 
Korelasi numerik antar fitur juga divisualisasikan menggunakan heatmap untuk pemahaman hubungan antar variabel.

## Data Preparation

1. Menghapus duplikasi: Menghilangkan baris data yang identik untuk mencegah bias model.
2. Mengisi missing value:
   - Fitur numerik diisi dengan median.
   -Fitur kategorikal diisi dengan modus.
3. Menghapus outlier (kuantil 1% dan 99%)
4. Feature engineering: `car_age = 2025 - Year`, hapus kolom `Year` karena sudah direpresentasikan oleh car_age
5. Pemisahan Data: Membagi data menjadi X dan y, lalu menjadi data latih (80%) dan data uji (20%).
6. Encoding & Scaling: `StandardScaler`, `OneHotEncoder`


### Contoh Kode: Feature Engineering
Potongan kode ini digunakan untuk membuat fitur baru car_age, yaitu menghitung umur mobil berdasarkan tahun saat ini dikurangi tahun mobil 
diproduksi (Year). Hal ini penting karena umur mobil merupakan salah satu faktor penting dalam menentukan harga. Setelah fitur baru dibuat, 
kolom Year dihapus karena informasinya telah terwakili.

Potongan kode berikut digunakan untuk membuat fitur baru bernama car_age yang merepresentasikan umur mobil berdasarkan tahun sekarang dikurangi 
tahun pembuatan mobil. Kemudian, kolom Year dihapus karena informasinya sudah diwakili oleh fitur car_age.

```python
CURRENT_YEAR = 2025
df['car_age'] = CURRENT_YEAR - df['Year']
df.drop('Year', axis=1, inplace=True)
```

### Contoh Kode: Pipeline Preprocessing
Kode ini mendefinisikan preprocessor yang menggabungkan dua jenis preprocessing:

-StandardScaler untuk menstandarisasi fitur numerik agar memiliki rata-rata 0 dan deviasi standar 1.
-OneHotEncoder untuk mengubah fitur kategorikal menjadi representasi numerik agar dapat digunakan oleh model machine learning.
 Semua proses ini dikombinasikan menggunakan ColumnTransformer.

Kode berikut mendefinisikan preprocessor yang menggunakan ColumnTransformer untuk memproses data secara otomatis. Kolom numerik 
akan diskalakan menggunakan StandardScaler untuk memastikan distribusi yang konsisten, sedangkan kolom kategorikal diubah menjadi 
format numerik melalui OneHotEncoder. Transformasi ini penting agar data dapat digunakan oleh model machine learning.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)
```
Pipeline preprocessing ini akan secara otomatis melakukan standardisasi dan encoding terhadap data input model.

## Modeling

### Algoritma:
Model yang digunakan dan cara kerjanya:

1. Random Forest Regressor
Merupakan ensemble dari beberapa pohon keputusan (decision trees) yang dilatih pada subset data secara acak.
Model ini mengambil rata-rata dari semua pohon untuk meminimalkan overfitting.
Parameter: n_estimators=100, random_state=42

2. Gradient Boosting Regressor
Model boosting yang membangun pohon keputusan secara bertahap dengan memfokuskan pada kesalahan prediksi sebelumnya.
Cocok untuk menangani data non-linear dan meningkatkan akurasi.
Parameter: n_estimators=100, random_state=42

3. MLP Regressor (Multi-layer Perceptron)
Merupakan jaringan saraf tiruan yang terdiri dari beberapa lapisan tersembunyi.
Dapat menangkap hubungan non-linear yang kompleks.
Parameter: hidden_layer_sizes=(64,32), max_iter=300, random_state=42
Setiap model diimplementasikan dalam pipeline untuk integrasi preprocessing dan training.

### Pipeline Random Forest
Pipeline ini menggabungkan preprocessing dan model Random Forest Regressor dalam satu alur. RandomForestRegressor bekerja dengan 
membangun banyak pohon keputusan secara acak dan menggabungkan hasilnya untuk membuat prediksi yang stabil dan akurat. 
Setelah model dilatih pada data latih, dilakukan prediksi pada data uji.

Kode ini membuat pipeline yang menyatukan proses preprocessing dan pelatihan model Random Forest. 
Pipeline terdiri dari dua tahap: preprocessing (scaling dan encoding) dan training menggunakan RandomForestRegressor. 
Model dilatih dengan data latih (X_train, y_train), lalu menghasilkan prediksi pada data uji (X_test).

Kode ini membuat pipeline yang menyatukan proses preprocessing dan pelatihan model. Pipeline terdiri dari dua tahap: 
preprocessing (scaling dan encoding) dan training menggunakan RandomForestRegressor. Model dilatih dengan data latih (X_train, y_train), 
lalu menghasilkan prediksi pada data uji (X_test).

```python
model_rf = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
```

### Pipeline Gradient Boosting
Pipeline ini menggabungkan preprocessing dan model GradientBoostingRegressor. Berbeda dengan Random Forest, 
Gradient Boosting membangun pohon keputusan secara bertahap, dengan setiap pohon baru memperbaiki kesalahan prediksi pohon sebelumnya. 
Ini memungkinkan model menangani kompleksitas data dengan lebih baik.

Kode berikut menyusun pipeline dengan model Gradient Boosting Regressor. Model ini bekerja dengan membangun pohon keputusan secara bertahap, 
di mana tiap pohon mencoba memperbaiki kesalahan dari pohon sebelumnya.

```python
model_gb = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
])
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)
```

### Pipeline MLP
Pipeline ini menggunakan MLPRegressor, model berbasis jaringan saraf tiruan (Artificial Neural Network). 
Dengan dua hidden layer berukuran 64 dan 32 neuron, model ini mampu mempelajari pola non-linear dari data. max_iter=300 
membatasi jumlah iterasi pelatihan agar proses training tidak berlangsung terlalu lama.

Berikut adalah pipeline untuk model jaringan saraf tiruan (MLP Regressor). Model ini terdiri dari dua hidden layer dengan 
jumlah neuron masing-masing 64 dan 32. MLP sangat baik untuk menangani hubungan non-linear dalam data.

```python
model_mlp = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=42))
])
model_mlp.fit(X_train, y_train)
y_pred_mlp = model_mlp.predict(X_test)
```

## Evaluation
Metrik Evaluasi:
-RMSE: Akar dari mean squared error, sensitif terhadap outlier.
-MAE: Rata-rata absolut kesalahan prediksi.
-R² Score: Proporsi varians yang bisa dijelaskan oleh model.

Contoh Potongan Kode: Evaluasi Model
-Fungsi evaluate() ini menghitung tiga metrik evaluasi utama:
-RMSE: Mengukur rata-rata kesalahan prediksi dalam satuan asli.
-MAE: Mengukur rata-rata kesalahan absolut.
-R² Score: Mengukur seberapa baik model menjelaskan variasi data target.
-Fungsi ini mencetak performa model dalam format yang rapi dan mudah dibaca.

Fungsi evaluate() digunakan untuk mengevaluasi performa model regresi dengan menghitung tiga metrik utama: RMSE, MAE, dan R² Score. 
RMSE menunjukkan rata-rata error kuadrat yang diakar, MAE memberikan rata-rata error absolut, dan R² Score mengukur proporsi variasi yang bisa dijelaskan oleh model. 
Fungsi ini mencetak hasil evaluasi dengan format yang rapi.

### Fungsi Evaluasi
```python
def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}:
  RMSE: {rmse:.2f}\n  MAE: {mae:.2f}\n  R2: {r2:.2f}\n")
```

### Hasil Evaluasi:
```
Random Forest:
  RMSE: 6063.87
  MAE: 2925.33
  R2: 0.97

Gradient Boosting:
  RMSE: 8582.63
  MAE: 5377.25
  R2: 0.94

MLP Regressor:
  RMSE: 5900.55
  MAE: 3648.21
  R2: 0.97
```

### Interpretasi:

-MLP Regressor memiliki RMSE terendah dan R² tertinggi, menunjukkan prediksi paling presisi.
-Random Forest unggul dalam MAE terendah, cocok bila ingin menghindari kesalahan absolut besar.
-Gradient Boosting sedikit di bawah dua lainnya dari sisi performa.

### Kesimpulan:
- MLP dan Random Forest sangat baik (akurasi tinggi)
- Random Forest unggul pada MAE (kesalahan absolut kecil)
- Gradient Boosting sedikit tertinggal
