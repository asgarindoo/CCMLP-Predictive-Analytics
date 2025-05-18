# Laporan Proyek Machine Learning - Asgarindo Dwiki Ibrahim Adji

## Domain Proyek
Obesitas merupakan salah satu masalah kesehatan utama yang semakin meningkat di seluruh dunia. Berdasarkan data dari Organisasi Kesehatan Dunia (WHO), obesitas didefinisikan sebagai kondisi medis di mana indeks massa tubuh (BMI) seseorang lebih besar dari 30 kg/m². Obesitas terkait dengan berbagai penyakit serius seperti diabetes tipe 2, hipertensi, penyakit jantung, dan beberapa jenis kanker. Peningkatan obesitas telah menjadi perhatian utama bagi kesehatan masyarakat global karena dampak jangka panjangnya terhadap kualitas hidup dan sistem kesehatan.

Menurut World Health Organization (WHO), prevalensi obesitas secara global telah meningkat hampir tiga kali lipat sejak tahun 1975. Pada tahun 2020, lebih dari 650 juta orang di seluruh dunia mengalami obesitas, yang berkontribusi pada lebih dari 4 juta kematian tahunan akibat penyakit yang terkait dengan obesitas. Angka ini diperkirakan akan terus meningkat jika tidak ada intervensi yang efektif dalam pencegahan dan pengelolaan obesitas.

Di Indonesia, berdasarkan data dari Riskesdas 2018, prevalensi obesitas pada orang dewasa mencapai 21,8%, dengan peningkatan signifikan dibandingkan tahun-tahun sebelumnya. Perubahan gaya hidup yang kurang aktif dan pola makan yang tidak sehat menjadi faktor utama yang berkontribusi pada peningkatan angka obesitas di negara ini.

Kenapa Masalah Ini Penting? obesitas bukan hanya masalah medis, tetapi juga beban sosial dan ekonomi yang besar. Oleh karena itu, sangat penting untuk mengidentifikasi individu yang berisiko mengalami obesitas melalui prediksi berbasis data, sehingga dapat dilakukan intervensi yang lebih cepat dan lebih tepat. Prediksi obesitas menggunakan teknik machine learning berdasarkan data demografis dan gaya hidup dapat membantu instansi kesehatan, perusahaan asuransi, dan individu dalam pengambilan keputusan yang berbasis data, serta memberikan edukasi pencegahan sejak dini. Sistem prediksi ini diharapkan dapat mengurangi risiko penyakit terkait obesitas, mengurangi beban sistem kesehatan, serta meningkatkan kualitas hidup masyarakat dengan mencegah obesitas secara lebih efektif.

- Kementerian Kesehatan Republik Indonesia, "Kurang Aktivitas Fisik Sebabkan Obesitas," Sehat Negeriku, 12-Jul-2023. [Online]. Available: https://sehatnegeriku.kemkes.go.id/baca/rilis-media/20230712/2043493/kurang-aktivitas-fisik-sebabkan-obesitas/.
- World Health Organization (WHO), "Obesity and Overweight," WHO, 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight.

## Business Understanding
### Problem Statements
- Bagaimana memprediksi tingkat obesitas seseorang berdasarkan data gaya hidup dan demografi?
- Bagaimana memastikan prediksi yang akurat terhadap tingkat obesitas agar bisa digunakan dalam pengambilan keputusan preventif oleh lembaga kesehatan atau individu?

### Goals
- Membangun model machine learning yang dapat memprediksi obesitas dengan akurasi tinggi
- Memastikan model memiliki performa yang baik menggunakan metrik evaluasi klasifikasi multi-kelas seperti akurasi, precision, recall, dan F1-score

    ### Solution statements
    - Menggunakan beberapa algoritma: Logistic Regression, Random Forest, dan KNN.
    - Melakukan preprocessing data yang menyeluruh, termasuk encoding, normalisasi, dan pemilihan fitur, guna      meningkatkan kualitas input model.
    - Menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1-score.

## Data Understanding
Dataset Prediksi Obesitas menyediakan kumpulan atribut yang komprehensif terkait demografi individu, kebiasaan gaya hidup, dan indikator kesehatan, yang bertujuan untuk memfasilitasi prediksi prevalensi obesitas. Dataset ini menawarkan sumber daya yang berharga bagi para peneliti, ilmuwan data, dan profesional kesehatan yang tertarik untuk mengeksplorasi interaksi kompleks dari faktor-faktor yang berkontribusi terhadap obesitas dan mengembangkan strategi intervensi yang efektif

- **Link**: [Kaggle - Obesity Prediction](https://www.kaggle.com/datasets/mrsimple07/obesity-prediction)
- **Jumlah data**: ±1000 baris
  
### Variabel-variabel pada Obesity Prediction dataset adalah sebagai berikut:
- `Age` (dalam tahun)
- `Gender` (pria atau wanita)
- `Height` (diukur dalam sentimeter atau inchi)
- `Weight` (diukur dalam kilogram atau pound)
- `BMI` (Metrik yang dihitung dari berat dan tinggi badan)
- `PhysicalActivityLevel` (Variabel ini mengukur tingkat aktivitas fisik)
- **Target**: `ObecityCategory` (tingkat obesitas: Underweight, Normalweight, Overweight, dll.)

### EDA & Visualisasi
Pada tahap Exploratory Data Analysis (EDA), dilakukan analisis untuk memahami struktur, pola, dan karakteristik dari dataset Obesity Prediction. Dataset terdiri dari 1000 entri dengan 7 kolom, yang mencakup fitur numerik seperti Age, Height, Weight, BMI, dan PhysicalActivityLevel, serta fitur kategorikal seperti Gender dan ObesityCategory. Data tidak memiliki nilai yang hilang atau duplikat, sehingga dapat dianggap bersih dari masalah tersebut. Namun, beberapa outlier ditemukan pada kolom Height, Weight, dan BMI, yang perlu ditangani lebih lanjut. Distribusi usia menunjukkan keragaman usia yang cukup besar, dengan sebagian besar data berada pada usia dewasa muda hingga paruh baya. Untuk distribusi BMI, sebagian besar individu berada dalam kategori normal, dengan beberapa di antaranya masuk dalam kategori obesitas atau underweight. Analisis distribusi kategori obesitas berdasarkan gender menunjukkan perbedaan yang cukup seimbang antara pria dan wanita, meskipun ada perbedaan kecil dalam prevalensi tiap kategori. Secara keseluruhan, EDA memberikan gambaran yang jelas tentang kualitas data dan pola-pola yang ada, serta memberikan dasar yang kuat untuk tahap preprocessing dan pengembangan model lebih lanjut

## Data Preparation
Pada bagian ini, diperlukan beberapa tahapan persiapan data untuk memastikan bahwa dataset siap digunakan dalam model machine learning.

- Pengecekan Missing Value dan Data Duplikat
  Pada tahap Exploratory Data Analysis (EDA) sebelumnya, dilakukan pengecekan untuk memastikan tidak ada missing value maupun data duplikat dalam dataset. Hasilnya menunjukkan bahwa tidak terdapat nilai yang hilang atau duplikat dalam dataset, sehingga masalah ini tidak perlu di tangani pada data preparation.
  - Missing Value: Tidak ditemukan nilai yang hilang pada kolom manapun dalam dataset.
  - Data Duplikat: Tidak ada data duplikat yang ditemukan

- Handling Outliers
  Pengecekan terhadap outliers dilakukan pada tahapan EDA dengan visualisasi menggunakan boxplot dan perhitungan statistik seperti Z-score atau IQR (Interquartile Range). Pada tahap ini outliers yang ditemukan dihapus atau disesuaikan nilainya agar tidak mempengaruhi model secara negatif.

- Encoding Fitur Kategorikal
    Fitur kategorikal dalam dataset, seperti Gender dan ObesityCategory, perlu diubah menjadi format numerik agar bisa diproses oleh model machine learning. Dalam hal ini, digunakan teknik Label Encoding, yang mengubah setiap kategori unik menjadi representasi angka. Teknik ini cocok untuk variabel kategorikal dengan urutan tertentu (seperti ObesityCategory), namun juga bisa digunakan untuk fitur lainnya.
    - Gender: Dikonversi menjadi dua label, Male menjadi 0 dan Female menjadi 1.
    - ObesityCategory: Dikonversi menjadi label numerik yang mewakili kategori obesitas, misalnya: Normalweight = 0, Obese = 1, Overweight = 2, dan Underweight = 3.

    Dengan encoding, model dapat lebih mudah mempelajari pola berdasarkan data kategorikal
  
  - Analisis Korelasi
    Analisis korelasi dilakukan untuk mengetahui sejauh mana hubungan antar fitur dalam dataset, khususnya antara fitur-fitur numerik dan target ObesityCategory. Tahapan ini dilakukan setelah proses encoding, karena fitur kategorikal yang telah diubah ke dalam format numerik akan lebih memungkinkan untuk dianalisis menggunakan teknik statistik seperti korelasi Pearson. Jika korelasi dilakukan sebelum encoding, maka fitur kategorikal yang masih berbentuk string tidak dapat diukur hubungannya secara statistik. Oleh karena itu, tahap encoding menjadi prasyarat penting sebelum melakukan analisis korelasi. Dalam hasilnya, meskipun terdapat beberapa fitur dengan korelasi rendah terhadap target, fitur-fitur tersebut tetap dipertahankan karena dianggap masih dapat memberikan kontribusi dalam model machine learning


- Pembagian Dataset
    Setelah proses encoding selesai, langkah selanjutnya adalah membagi dataset menjadi dua bagian, data pelatihan dan data pengujian. Pembagian ini dilakukan menggunakan fungsi train_test_split dari library sklearn.model_selection untuk melakukan pembagian. Pembagian dilakukan dengan perbandingan 80:20, yang berarti 80% data digunakan untuk pelatihan model dan 20% untuk pengujian.

    Pembagian ini dilakukan untuk menguji kinerjanya pada data yang belum pernah dilihat sebelumnya

- Standarisasi Data
    Untuk memastikan bahwa semua fitur berada dalam skala yang sama, dilakukan proses standarisasi pada data numerik. Proses ini menggunakan StandardScaler dari sklearn.preprocessing, yang akan mengubah setiap fitur sehingga memiliki rata-rata 0 dan deviasi standar 1.

    Standarisasi ini penting agar algoritma machine learning seperti Decision Tree, Random Forest, dan Logistic Regression dapat bekerja lebih baik dengan mempertimbangkan seluruh fitur secara adil.

## Modeling
Pada tahap ini, diterapkan tiga algoritma machine learning untuk memprediksi obesitas berdasarkan data gaya hidup dan demografi, yaitu **Logistic Regression**, **Random Forest**, dan **K-Nearest Neighbors (KNN)**. Masing-masing algoritma dijalankan dengan parameter default, kecuali untuk KNN yang dikonfigurasi menggunakan 10 tetangga. Berikut adalah penjelasan mengenai masing-masing algoritma yang digunakan:

### 1. **Logistic Regression**
   - **Penjelasan**:
     - Logistic Regression adalah algoritma klasifikasi yang digunakan untuk memprediksi probabilitas suatu kelas. Ini adalah model yang sering digunakan untuk masalah klasifikasi biner, namun juga dapat digunakan untuk klasifikasi multi-kelas.
   - **Parameter yang digunakan**:
     - Menggunakan parameter default:
       - `solver='liblinear'`: Algoritma optimisasi yang digunakan adalah `liblinear`, yang cocok untuk dataset kecil dan sedang.
       - `max_iter=100`: Jumlah iterasi maksimum yang dilakukan dalam proses pelatihan.
   - **Kelebihan**:
     - Cepat dan mudah diinterpretasikan.
     - Sangat efisien untuk dataset kecil dan sederhana.
   - **Kekurangan**:
     - Tidak cocok untuk data yang memiliki hubungan non-linear.
     - Kurang fleksibel jika datasetnya sangat kompleks.

### 2. **Random Forest**
   - **Penjelasan**:
     - Random Forest adalah algoritma ensemble yang membangun beberapa pohon keputusan dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting.
   - **Parameter yang digunakan**:
     - Menggunakan parameter default:
       - `n_estimators=100`: Jumlah pohon keputusan dalam hutan. Dengan 100 pohon, model ini dapat menangani variabilitas data dengan baik.
       - `max_depth=None`: Pohon akan tumbuh tanpa batas kedalaman, memungkinkan model menangkap hubungan kompleks dalam data.
       - `random_state=42`: Untuk memastikan hasil yang konsisten pada setiap percakapan model.
   - **Kelebihan**:
     - Dapat menangani data yang sangat besar dan bervariasi dengan baik.
     - Mengurangi risiko overfitting dibandingkan dengan pohon keputusan tunggal.
   - **Kekurangan**:
     - Proses pelatihan dan prediksi lebih lambat dibandingkan dengan model lain seperti Logistic Regression.
     - Model lebih sulit untuk diinterpretasikan.

### 3. **K-Nearest Neighbors (KNN)**
   - **Penjelasan**:
     - KNN adalah algoritma yang mengklasifikasikan data berdasarkan kedekatannya dengan tetangga-tetangga terdekat di data pelatihan. KNN adalah model non-parametrik yang tidak memerlukan pelatihan secara eksplisit.
   - **Parameter yang digunakan**:
     - `n_neighbors=10`: Menggunakan 10 tetangga terdekat untuk menentukan kelas dari data yang ingin diprediksi.
     - `weights='uniform'`: Setiap tetangga memiliki bobot yang sama dalam menentukan kelas.
   - **Kelebihan**:
     - Mudah dipahami dan diimplementasikan.
     - Tidak memerlukan pelatihan model secara eksplisit.
   - **Kekurangan**:
     - Tidak efisien pada dataset besar karena harus menghitung jarak antara setiap data untuk setiap prediksi.
     - Sensitif terhadap skala fitur, sehingga standarisasi data sangat penting.

### Pemilihan Model Terbaik: **Random Forest**
Setelah mengevaluasi ketiga model, **Random Forest** memberikan performa terbaik dengan akurasi tertinggi, meskipun tanpa tuning hyperparameter lebih lanjut. Oleh karena itu, Random Forest dipilih sebagai model terbaik untuk memprediksi obesitas pada dataset ini. Meskipun Logistic Regression juga memberikan hasil yang baik, Random Forest memberikan akurasi yang lebih stabil dan mampu menangani data yang lebih kompleks.

## Evaluation
Setelah melakukan serangkaian eksperimen menggunakan tiga algoritma yaitu Logistic Regression, Random Forest, dan K-Nearest Neighbors (KNN). Hasil evaluasi menunjukkan bahwa Random Forest memberikan performa terbaik untuk tugas ini. Dengan akurasi 98.97% serta nilai precision, recall, dan F1-score yang sangat tinggi, model ini telah berhasil menjawab problem statement utama proyek ini, yaitu bagaimana memprediksi tingkat obesitas seseorang berdasarkan data gaya hidup dan demografi mereka. Secara keseluruhan, proyek ini berhasil mencapainya semua tujuan yang ditetapkan dalam Business Understanding.

Apakah sudah menjawab setiap problem statement?
Ya, proyek ini telah berhasil memprediksi tingkat obesitas dengan akurasi yang sangat tinggi. Dengan memanfaatkan data gaya hidup dan demografi, model yang dibangun secara efektif mengklasifikasikan individu ke dalam kategori obesitas yang sesuai, menjawab problem statement yang berfokus pada prediksi tingkat obesitas.

Apakah berhasil mencapai setiap goals yang diharapkan?
Proyek ini telah mencapai setiap goal yang telah ditetapkan:
  - Membangun model prediksi obesitas berbasis data: Berhasil, dengan model Random Forest yang mampu memberikan akurasi yang sangat tinggi.
  - Menghasilkan model dengan akurasi tinggi untuk klasifikasi multi-kelas: Dicapai dengan hasil evaluasi model yang menunjukkan akurasi 98.97% dan performa yang sangat baik dalam hal precision, recall, dan F1-score.

Apakah setiap solusi statement yang direncanakan berdampak?
Solusi yang dirancang memiliki dampak yang signifikan dalam konteks pencegahan dan deteksi dini obesitas. Model ini memiliki potensi aplikasi nyata dalam beberapa area, antara lain:
  -  Model ini dapat membantu profesional kesehatan untuk mengidentifikasi individu dengan risiko obesitas, memungkinkan intervensi lebih cepat.
  - Model ini dapat digunakan oleh perusahaan teknologi kesehatan untuk mengembangkan layanan yang lebih personal, seperti rekomendasi diet, olahraga, dan manajemen kesehatan berbasis kategori obesitas.
  - Aplikasi atau sistem yang menggunakan model ini bisa memberikan rekomendasi gaya hidup sehat secara lebih tepat, berfokus pada individu yang lebih membutuhkan bantuan dalam mengelola kesehatannya.


### Kesimpulan
Berdasarkan semua metrik evaluasi, Random Forest adalah model terbaik yang berhasil mencapai tujuan yang ditetapkan dalam Business Understanding. Model ini menunjukkan akurasi tinggi dan performa yang sangat baik dalam klasifikasi multi-kelas, menjadikannya pilihan utama untuk memprediksi tingkat obesitas. Solusi yang dihasilkan memberikan dampak positif dalam konteks pencegahan obesitas, deteksi dini, serta pengambilan keputusan yang dapat diimplementasikan oleh lembaga kesehatan atau perusahaan kesehatan digital.