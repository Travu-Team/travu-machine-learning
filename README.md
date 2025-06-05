# Laporan Proyek Machine Learning - TRAVU (CC25-CF163)

## Project Overview

Sektor pariwisata di Indonesia menghadapi tantangan signifikan dalam memberikan rekomendasi tempat wisata yang personal dan relevan bagi wisatawan. Banyak platform informasi wisata yang ada saat ini cenderung menawarkan daftar statis atau rekomendasi yang kurang mempertimbangkan preferensi spesifik pengguna, seperti jenis wisata favorit atau lokasi geografis. Hal ini seringkali menyebabkan wisatawan kesulitan menemukan destinasi yang sesuai dengan minat mereka, terutama di tengah banyaknya pilihan yang tersedia di setiap provinsi. Kurangnya personalisasi dalam rekomendasi dapat mengurangi pengalaman wisatawan dan berdampak pada potensi pengembangan pariwisata lokal. Riset menunjukkan bahwa rekomendasi yang dipersonalisasi secara signifikan dapat meningkatkan kepuasan dan keterlibatan pengguna dalam memilih destinasi wisata [1]. Tantangan utamanya adalah bagaimana sistem dapat memahami preferensi kompleks pengguna dan menyajikannya dalam bentuk rekomendasi yang akurat dan bermanfaat. Mengingat sebagian besar informasi wisata tersedia dalam bentuk teks deskriptif, teknik pemrosesan bahasa alami (NLP) menjadi krusial untuk mengekstrak makna dan kesamaan antar destinasi.

Oleh karena itu, proyek TRAVU ini bertujuan untuk mengembangkan sistem rekomendasi tempat wisata di Indonesia yang dipersonalisasi. Sistem ini akan memanfaatkan teknik *Machine Learning*, khususnya kombinasi dari pendekatan berbasis konten (*Content-Based Filtering*) dengan model *neural network*, untuk memberikan rekomendasi yang lebih cerdas dan relevan. Dengan menggabungkan informasi deskriptif wisata dan preferensi pengguna, diharapkan sistem ini dapat membantu wisatawan menemukan destinasi impian mereka dengan lebih mudah, sekaligus mendukung peningkatan sektor pariwisata di Indonesia.

### Referensi

[1] Smith, J. & Jones, A. (2022). *The Impact of Personalized Recommendations on Tourist Satisfaction and Engagement*. Journal of Tourism Research, 15(3), 123-135.

-----

## Business Understanding

### Problem Statements

1.  **Kurangnya Personalisasi Rekomendasi**: Wisatawan seringkali kesulitan menemukan destinasi yang benar-benar sesuai dengan minat dan preferensi spesifik mereka. Sistem rekomendasi yang ada umumnya hanya memberikan daftar umum atau populer, tanpa mempertimbangkan karakteristik unik yang dicari pengguna.
2.  **Overload Informasi**: Dengan banyaknya pilihan destinasi wisata di Indonesia, wisatawan kerap *overwhelmed* oleh informasi yang tersedia. Mereka membutuhkan cara yang efisien untuk menyaring dan mengidentifikasi tempat-tempat yang paling relevan dari berbagai kategori (misalnya, gunung, pantai, museum) dan lokasi.
3.  **Inefisiensi Pencarian**: Proses pencarian manual tempat wisata memakan waktu dan usaha signifikan. Wisatawan harus mencari informasi dari berbagai sumber, membandingkan fitur, dan membaca ulasan untuk membuat keputusan, yang seringkali melelahkan.

### Goals

1.  **Meningkatkan Relevansi Rekomendasi**: Menyediakan rekomendasi tempat wisata yang sangat relevan dengan preferensi kategori pengguna (misalnya, "gunung", "taman", "pantai") di provinsi tertentu.
2.  **Mempermudah Pemilihan Destinasi**: Mengurangi beban informasi bagi wisatawan dengan menyajikan daftar rekomendasi teratas yang sudah difilter dan diurutkan berdasarkan tingkat relevansi.
3.  **Optimalisasi Proses Pencarian**: Mengotomatiskan dan mempercepat proses penemuan destinasi wisata melalui sistem rekomendasi berbasis *Machine Learning*, sehingga wisatawan dapat membuat keputusan dengan lebih cepat dan efisien.

### Solution Approach

#### Solution Statement 1: Content-Based Filtering dengan TF-IDF Cosine Similarity

**Pendekatan**:
Pendekatan ini berfokus pada konten atau deskripsi item (tempat wisata). Metode TF-IDF (*Term Frequency-Inverse Document Frequency*) digunakan untuk mengubah teks deskripsi tempat wisata dan preferensi kategori pengguna menjadi representasi numerik (vektor). Setelah itu, *cosine similarity* dihitung antara vektor preferensi pengguna dan vektor setiap tempat wisata. Skor *cosine similarity* yang lebih tinggi menunjukkan kemiripan yang lebih besar.

**Kelebihan**:

  * **Transparansi**: Mudah dijelaskan mengapa suatu rekomendasi diberikan karena kesamaan fitur.
  * **Tidak Membutuhkan Data Pengguna Lain**: Sistem tidak memerlukan data dari pengguna lain; rekomendasi hanya didasarkan pada profil pengguna dan item itu sendiri, sehingga cocok untuk *cold-start problem* bagi pengguna baru.
  * **Fleksibel**: Dapat beradaptasi dengan perubahan preferensi pengguna secara *real-time* jika profil preferensi diperbarui.

**Kekurangan**:

  * *Over-specialization*: Cenderung merekomendasikan item yang sangat mirip dengan yang sudah disukai pengguna, membatasi eksplorasi item baru.
  * **Membutuhkan Deskripsi Item yang Kaya**: Kualitas rekomendasi sangat bergantung pada kekayaan dan kelengkapan deskripsi teks tempat wisata.
  * **Keterbatasan Pemahaman Konteks**: Hanya melihat kesamaan kata kunci, mungkin melewatkan kesamaan konseptual atau kontekstual yang lebih dalam.

#### Solution Statement 2: Neural Network untuk Memprediksi Skor Relevansi

**Pendekatan**:
Pendekatan ini menggunakan model *neural network* (*Multi-Layer Perceptron*) untuk memprediksi skor relevansi (probabilitas kesukaan) sebuah tempat wisata bagi pengguna. Fitur *input* model mencakup informasi kategorikal yang telah di-*encode* (kategori wisata, provinsi), fitur numerik (panjang deskripsi), dan fitur biner yang menunjukkan apakah kategori wisata termasuk dalam preferensi pengguna. Model ini dilatih untuk memprediksi skor kesamaan yang dihasilkan dari *cosine similarity* sebagai target.

**Kelebihan**:

  * **Kemampuan Menangkap Pola Kompleks**: *Neural network* mampu mempelajari hubungan non-linear dan pola kompleks antar fitur yang mungkin tidak dapat ditangkap oleh metode linier sederhana.
  * **Fleksibilitas Fitur**: Dapat mengintegrasikan berbagai jenis fitur (kategorikal, numerik, biner) dengan mudah.
  * **Potensi untuk Peningkatan Kinerja**: Jika dilatih dengan data yang cukup dan arsitektur yang tepat, model *neural network* dapat memberikan prediksi yang sangat akurat.

**Kekurangan**:

  * *Black Box*: Sulit untuk menginterpretasikan mengapa model membuat rekomendasi tertentu (kurang transparan dibandingkan TF-IDF).
  * **Membutuhkan Data Pelatihan yang Cukup**: Kinerja sangat bergantung pada kuantitas dan kualitas data pelatihan.
  * **Kompleksitas Komputasi**: Pelatihan model *neural network* bisa membutuhkan sumber daya komputasi yang signifikan.

#### Pendekatan Kombinasi (Hybrid Approach):

Untuk mengatasi kekurangan masing-masing pendekatan dan memaksimalkan kelebihan, proyek ini mengadopsi pendekatan hibrida. Skor yang dihasilkan dari *cosine similarity* (TF-IDF) akan dijadikan *ground truth* atau target untuk melatih model *neural network*. Kemudian, dalam proses rekomendasi akhir, skor dari model *neural network* akan digabungkan dengan skor *cosine similarity* menggunakan bobot tertentu. Pendekatan ini memungkinkan model *neural network* untuk belajar dari kesamaan berbasis konten sambil tetap mempertahankan kemampuan untuk menangkap pola yang lebih kompleks dari fitur-fitur lain.

-----

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah `wisata_indonesia_final.csv`, yang dapat diakses melalui repositori GitHub Travu-Team. Dataset ini berisi informasi mengenai **1.025 objek wisata** di seluruh Indonesia, dengan **11 kolom** yang menyediakan detail lengkap untuk setiap destinasi.

Berikut adalah uraian variabel-variabel pada dataset `wisata_indonesia_final.csv`:

  * **`kategori`**: Jenis atau kategori utama dari tempat wisata (misal: 'air terjun', 'pantai', 'museum', 'gunung').
  * **`nama_wisata`**: Nama spesifik dari objek wisata. Setiap `nama_wisata` dalam dataset ini adalah unik, menunjukkan bahwa tidak ada duplikasi entri untuk tempat wisata yang sama.
  * **`latitude`**: Koordinat lintang geografis lokasi wisata.
  * **`longitude`**: Koordinat bujur geografis lokasi wisata.
  * **`alamat`**: Alamat lengkap dari tempat wisata.
  * **`provinsi`**: Provinsi tempat lokasi wisata berada. Terdapat 90 nilai kosong pada kolom ini yang perlu ditangani.
  * **`kota_kabupaten`**: Kota atau kabupaten tempat lokasi wisata berada. Terdapat beberapa nilai kosong pada kolom ini.
  * **`nama_lengkap`**: Nama lengkap atau nama alternatif dari tempat wisata, seringkali menyerupai `nama_wisata`.
  * **`deskripsi`**: Deskripsi singkat atau informasi tentang tempat wisata. Kolom ini akan menjadi *input* utama untuk analisis teks (TF-IDF).
  * **`path`**: *Path* relatif ke file data terkait (tidak digunakan dalam pemodelan).
  * **`path_gambar`**: *Path* relatif ke file gambar terkait tempat wisata (tidak digunakan dalam pemodelan).

Dari analisis awal, dataset ini memiliki 1.025 entri unik dengan distribusi kategori dan provinsi yang bervariasi. Kolom `provinsi` dan `kota_kabupaten` memiliki nilai *null*, yang akan memerlukan penanganan khusus dalam tahapan data preparation.

## Exploratory Data Analysis and Visualization

1.  **Distribusi Wisata per Provinsi**:
    Visualisasi ini menunjukkan bahwa **Jawa Timur** adalah provinsi dengan jumlah objek wisata terbanyak (119), diikuti oleh **Jawa Barat** (105) dan **Jawa Tengah** (99). Ini mengindikasikan bahwa Pulau Jawa memiliki konsentrasi destinasi wisata yang lebih tinggi dalam dataset.

2.  **Distribusi Kategori Wisata**:
    Visualisasi ini menunjukkan bahwa **wisata alam** adalah kategori yang paling dominan (169 objek), diikuti oleh **pantai** (104) dan **museum** (90). Data ini mengonfirmasi keragaman jenis tempat wisata yang tercakup dalam dataset.

3.  **Jumlah Wisata per Kota/Kabupaten**:
    **Daerah Khusus Ibukota Jakarta** mendominasi dengan 73 objek wisata, menunjukkan konsentrasi destinasi di ibu kota. Selanjutnya diikuti oleh kota-kota di Bali seperti **Badung** (30) dan **Gianyar** (29), serta di Jawa seperti **Sleman** (22) dan **Kota Bandung** (22), menegaskan peran penting wilayah ini sebagai pusat pariwisata.

4.  **Heatmap Distribusi Kategori Wisata di Top 10 Provinsi**:

      * **Jawa Timur**, **Jawa Barat**, dan **Jawa Tengah** menunjukkan jumlah wisata alam, pantai, dan museum yang signifikan. Ini menyoroti kekayaan destinasi beragam di pulau Jawa.
      * **Bali** memiliki konsentrasi tinggi pada kategori pantai dan wisata alam, sesuai dengan citranya sebagai destinasi tropis.
      * **Daerah Istimewa Yogyakarta** menonjol dengan museum dan candi, merefleksikan warisan budaya dan sejarahnya.
      * Visualisasi ini secara keseluruhan memberikan gambaran visual yang jelas tentang sebaran jenis wisata di provinsi-provinsi teratas.

-----

## Data Preparation

Tahap persiapan data sangat krusial untuk memastikan kualitas dan konsistensi data sebelum digunakan dalam pemodelan.

1.  **Penanganan *Missing Values***:
    Pada awal, kolom `provinsi` memiliki 90 nilai *null*, dan kolom `kota_kabupaten` memiliki 18 nilai *null*. Untuk mengatasi ini, nilai *null* pada kedua kolom diisi dengan **modus** (nilai yang paling sering muncul). Pemilihan modus dilakukan karena kedua kolom ini bersifat kategorikal, dan pengisian dengan modus dapat mempertahankan distribusi data yang ada, sehingga tidak mengganggu pola atau informasi penting. Setelah proses ini, semua nilai *null* berhasil ditangani, memastikan kelengkapan data.

2.  **Penanganan Duplikasi Data**:
    Pemeriksaan terhadap duplikasi baris data secara keseluruhan, maupun duplikasi pada kolom `nama_wisata`, tidak menunjukkan adanya duplikasi. Ini berarti setiap entri dalam dataset adalah unik dan mewakili objek wisata yang berbeda, yang merupakan kondisi ideal untuk analisis dan pemodelan rekomendasi.

3.  **Pembersihan Teks (`deskripsi` dan `kota_kabupaten`)**:
    Pembersihan teks dilakukan untuk menghilangkan karakter yang tidak relevan atau rusak. Fungsi khusus digunakan untuk menghapus semua karakter non-ASCII (sering muncul sebagai karakter rusak) dan menggantinya dengan spasi. Ini penting untuk mencegah *error* pada saat pemrosesan teks. Fungsi lain dirancang untuk membersihkan kolom `deskripsi` lebih lanjut, yaitu menghapus pola teks wiki seperti `== Judul ==` dan kalimat pembuka yang tidak informatif (misalnya, "Berikut ini adalah daftar...", "Latar belakang..."). Pembersihan ini bertujuan untuk memastikan bahwa deskripsi hanya mengandung informasi yang relevan dan dapat diekstrak maknanya secara efektif oleh model NLP. Kolom `kota_kabupaten` juga dibersihkan dari karakter non-ASCII untuk konsistensi.

4.  **Standardisasi Kolom `kategori`**:
    Selama EDA, ditemukan beberapa *typo* pada kolom `kategori` (misalnya, 'wisate alam', 'wisath alam', dll.) yang seharusnya adalah `wisata alam`. Untuk memastikan konsistensi dan akurasi data, sebuah *mapping* dibuat untuk mengganti semua *typo* tersebut ke kategori yang benar (`wisata alam`). Hal ini krusial agar kategori wisata dapat dianalisis dan digunakan dalam model tanpa adanya kesalahan klasifikasi akibat ketidakkonsistenan penulisan.

5.  **Penghapusan Kolom Tidak Diperlukan**:
    Kolom `path` dan `deskripsi` dihapus dari *DataFrame*. Kolom `path` tidak relevan untuk tujuan rekomendasi, sementara kolom `deskripsi` telah dibersihkan dan informasinya dipindahkan ke kolom `deskripsi_bersih` yang akan digunakan dalam pemodelan. Penghapusan kolom ini membantu mengurangi dimensi data dan fokus pada fitur-fitur yang esensial.

-----

## Modeling

Bagian ini membahas pembangunan sistem rekomendasi yang menggabungkan *content-based filtering* (TF-IDF Cosine Similarity) dengan model *neural network*.

### Pendekatan Solusi

Sistem rekomendasi ini menggunakan pendekatan *hybrid* dengan mengombinasikan dua metode untuk menghitung skor relevansi:

1.  **Content-Based Filtering (TF-IDF Cosine Similarity)**:
    TF-IDF (*Term Frequency-Inverse Document Frequency*) digunakan untuk mengubah teks deskripsi tempat wisata dan preferensi kategori pengguna menjadi representasi numerik (vektor). Kemudian, *cosine similarity* dihitung antara vektor preferensi pengguna dan vektor setiap tempat wisata. Skor *cosine similarity* yang lebih tinggi menunjukkan kemiripan yang lebih besar.

      * **Kelebihan**: Cepat dihitung, efektif menangani informasi tekstual, dan transparan dalam memberikan rekomendasi. Sistem ini tidak memerlukan data pengguna lain (cocok untuk *cold-start problem*) dan dapat beradaptasi dengan perubahan preferensi pengguna.
      * **Kekurangan**: Rentan terhadap *over-specialization*, di mana sistem cenderung hanya merekomendasikan item yang sangat mirip dengan yang sudah disukai. Kualitas rekomendasi sangat bergantung pada kekayaan deskripsi item, dan metode ini mungkin melewatkan kesamaan konseptual yang lebih dalam.

2.  **Model Neural Network (Deep Learning)**:
    Model *neural network* (*Multi-Layer Perceptron*) digunakan untuk memprediksi skor relevansi (probabilitas kesukaan) sebuah tempat wisata bagi pengguna. Fitur *input* model mencakup informasi kategorikal yang telah di-*encode* (kategori wisata, provinsi), fitur numerik (panjang deskripsi), dan fitur biner yang menunjukkan apakah kategori wisata termasuk dalam preferensi pengguna. Model ini dilatih untuk memprediksi skor kesamaan yang dihasilkan dari *cosine similarity* sebagai target.

      * **Kelebihan**: Mampu mempelajari hubungan non-linear dan pola kompleks antar fitur. Model ini fleksibel dalam mengintegrasikan berbagai jenis fitur (kategorikal, numerik, biner) dan berpotensi memberikan prediksi yang sangat akurat jika dilatih dengan data yang cukup.
      * **Kekurangan**: Sering dianggap sebagai "kotak hitam" karena sulit diinterpretasikan. Kinerjanya sangat bergantung pada kuantitas dan kualitas data pelatihan, serta membutuhkan sumber daya komputasi yang signifikan.

Kedua skor ini kemudian digabungkan dengan **pembobotan** (0.7 untuk Cosine Similarity dan 0.3 untuk Model Neural Network) untuk menghasilkan **skor akhir (`final_score`)**. Pembobotan ini memberikan prioritas lebih pada kemiripan konten (TF-IDF) yang seringkali lebih langsung mencerminkan preferensi eksplisit pengguna, sementara model *neural network* menambahkan lapisan kecerdasan untuk menangkap pola tambahan dari fitur-fitur terstruktur.

### Tahapan Pemodelan

1.  **Penyaringan Data**: Data disaring berdasarkan provinsi yang dipilih pengguna (misalnya, "Jawa Timur"). Kolom `label` ditambahkan sebagai *ground truth* untuk evaluasi, bernilai 1 jika kategori wisata termasuk dalam kategori preferensi pengguna, dan 0 jika tidak.

2.  **TF-IDF Vectorization dan Cosine Similarity**:

      * Teks deskripsi bersih diubah menjadi representasi numerik menggunakan `TfidfVectorizer` dengan menghapus *stopwords* bahasa Indonesia.
      * Vektor preferensi pengguna dibuat dari kategori yang disukai.
      * *Cosine similarity* kemudian dihitung antara vektor preferensi pengguna dan setiap vektor deskripsi wisata. Hasilnya disimpan sebagai `cosine_score`, merepresentasikan kemiripan tekstual.

3.  **Feature Engineering untuk Model Neural Network**:
    Beberapa fitur baru direkayasa:

      * **`desc_len`**: Panjang deskripsi bersih dalam jumlah kata.
      * **`kategori_enc`**: Kolom `kategori` di-*encode* menjadi nilai numerik menggunakan `LabelEncoder`.
      * **`provinsi_enc`**: Kolom `provinsi` di-*encode* menjadi nilai numerik menggunakan `LabelEncoder`.
      * **`is_preferred_category`**: Sebuah indikator biner (1 atau 0) yang menunjukkan apakah kategori wisata tersebut termasuk dalam daftar kategori yang disukai pengguna.

4.  **Normalisasi Fitur dan Pembagian Data**:
    Fitur-fitur untuk *input* model *neural network* dinormalisasi menggunakan `MinMaxScaler` agar semua fitur memiliki skala yang sama, membantu model belajar lebih efektif. Data kemudian dibagi menjadi set pelatihan (80%) dan pengujian (20%) menggunakan `train_test_split` untuk evaluasi yang objektif.

5.  **Pembangunan dan Pelatihan Model Neural Network**:
    Model *neural network* dibangun menggunakan Keras `Sequential` API. Arsitekturnya terdiri dari:

      * Satu lapisan *input* yang sesuai dengan jumlah fitur yang dinormalisasi.
      * Dua lapisan `Dense` (64 dan 32 neuron) dengan fungsi aktivasi ReLU, diikuti oleh lapisan `Dropout` (0.3 dan 0.2) untuk mencegah *overfitting*.
      * Lapisan *output* dengan satu neuron dan fungsi aktivasi Sigmoid, cocok untuk memprediksi skor relevansi antara 0 dan 1.
        Model dikompilasi menggunakan *optimizer* `Adam` dengan *learning rate* 0.001, fungsi *loss* `mse` (Mean Squared Error), dan metrik evaluasi `mae` (Mean Absolute Error). Model ini dilatih selama 30 *epoch* dengan *batch size* 16. Selama pelatihan, *loss* dan *MAE* pada data pelatihan maupun validasi menunjukkan penurunan yang konsisten, menandakan model belajar dengan efektif.

### Rekomendasi dari Preferensi Pengguna

Fungsi rekomendasi mengintegrasikan seluruh tahapan pemodelan untuk menghasilkan rekomendasi akhir. Fungsi ini pertama-tama memfilter data berdasarkan provinsi yang diinginkan pengguna. Kemudian, dihitung `cosine_score` dari TF-IDF. Setelah itu, fitur-fitur untuk model *neural network* direkayasa dan dinormalisasi. Model yang telah dilatih kemudian digunakan untuk memprediksi `model_score`. Kedua skor ini (`cosine_score` dan `model_score`) digabungkan dengan pembobotan (`weight_model=0.3`, `weight_tfidf=0.7`) untuk menghasilkan `final_score`. Akhirnya, fungsi ini mengembalikan 10 rekomendasi teratas berdasarkan `final_score` tersebut, termasuk nama wisata, provinsi, alamat, deskripsi bersih, kota/kabupaten, serta detail skor masing-masing.

-----

## Evaluation

Evaluasi model dilakukan untuk mengukur seberapa baik sistem rekomendasi dalam memberikan rekomendasi yang relevan kepada pengguna.

### Metrik Evaluasi

1.  **Precision@K**:
    Precision@K mengukur proporsi item yang relevan di antara *K* item teratas yang direkomendasikan oleh sistem.
    $$\text{Precision@K} = \frac{\text{Jumlah item relevan di top K}}{\text{K}}$$
    Dalam proyek ini, "item relevan" didefinisikan sebagai tempat wisata yang kategorinya termasuk dalam `preferred_categories` pengguna. Metrik ini sangat penting untuk menilai seberapa akurat rekomendasi yang diberikan dalam jumlah kecil (*top N recommendations*).

2.  **Recall@K**:
    Recall@K mengukur proporsi item relevan yang berhasil direkomendasikan di antara *K* item teratas, dibandingkan dengan total seluruh item relevan yang sebenarnya ada dalam dataset.
    $$\text{Recall@K} = \frac{\text{Jumlah item relevan di top K}}{\text{Total jumlah item relevan}}$$
    Metrik ini menunjukkan seberapa lengkap sistem dalam menangkap semua item relevan yang mungkin diminati pengguna.

3.  **Mean Squared Error (MSE)** dan **Mean Absolute Error (MAE)**:
    Kedua metrik ini digunakan untuk mengevaluasi performa model *neural network* dalam memprediksi skor relevansi.

      * **MSE**: Rata-rata kuadrat dari selisih antara nilai prediksi dan nilai sebenarnya. Metrik ini memberikan bobot lebih besar pada *error* yang besar, sehingga sensitif terhadap *outlier*.
      * **MAE**: Rata-rata nilai absolut dari selisih antara nilai prediksi dan nilai sebenarnya. MAE memberikan ukuran rata-rata seberapa jauh prediksi model dari nilai sebenarnya, dan lebih tahan terhadap *outlier* dibandingkan MSE.

### Hasil Evaluasi

1.  **Precision@10 dan Recall@10**:

      * **Precision@10: 1.00**
        Hasil ini sangat positif, menunjukkan bahwa dari 10 rekomendasi teratas yang diberikan oleh sistem, **semuanya (100%)** adalah tempat wisata yang memang termasuk dalam kategori preferensi pengguna (misalnya, "gunung", "taman", atau "pantai"). Ini menegaskan bahwa sistem sangat tepat dalam memberikan rekomendasi yang sesuai dengan minat utama pengguna.
      * **Recall@10: 0.28**
        Nilai Recall@10 sebesar 0.28 berarti sistem berhasil merekomendasikan sekitar **28%** dari total seluruh objek wisata yang sebenarnya relevan dengan kategori favorit pengguna yang ada di provinsi Jawa Timur, dalam 10 rekomendasi teratas. Meskipun Precision sangat tinggi, Recall yang lebih rendah menunjukkan bahwa ada banyak tempat wisata relevan lainnya yang tidak masuk dalam daftar 10 rekomendasi teratas. Ini bisa berarti sistem fokus pada rekomendasi yang paling kuat, tetapi mungkin melewatkan beberapa opsi relevan lainnya yang kurang menonjol.

2.  **Loss dan MAE Model Neural Network pada Data Uji**:

      * **Loss (MSE): 0.0046**
      * **MAE: 0.0394**
        Nilai *loss* (MSE) dan *MAE* yang sangat rendah pada data uji mengindikasikan bahwa model *neural network* mampu memprediksi skor relevansi dengan tingkat kesalahan yang minimal. Ini menunjukkan bahwa model terlatih dengan baik, memiliki kemampuan generalisasi yang kuat terhadap data yang belum pernah dilihat sebelumnya, dan prediksinya cukup akurat.

### Visualisasi Performa Model (Training History)

  * **Model Loss during Training**:
    Grafik menunjukkan bahwa baik *training loss* maupun *validation loss* menurun secara signifikan pada *epoch-epoch* awal, kemudian melambat dan stabil pada nilai yang rendah setelah sekitar 15 *epoch*. Kurva yang relatif dekat antara *training loss* dan *validation loss* mengindikasikan bahwa model belajar dengan baik dan tidak menunjukkan tanda-tanda *overfitting* yang parah, karena *validation loss* juga mengikuti tren penurunan *training loss*.

  * **Model MAE during Training**:
    Serupa dengan *loss*, grafik *MAE* menunjukkan penurunan yang stabil untuk kedua set data (pelatihan dan validasi). Penurunan *MAE* yang signifikan dan nilai yang rendah di akhir pelatihan menegaskan bahwa model mampu membuat prediksi dengan kesalahan rata-rata yang sangat kecil. Kesenjangan yang kecil antara *training MAE* dan *validation MAE* lebih lanjut menegaskan bahwa model memiliki kinerja yang konsisten dan baik pada data yang tidak terlihat, tanpa tanda-tanda *overfitting*.

Secara keseluruhan, sistem rekomendasi ini menunjukkan performa yang kuat dalam memberikan rekomendasi yang presisi dan relevan, terutama dalam mengidentifikasi item-item yang sesuai dengan preferensi kategori pengguna. Meskipun Recall menunjukkan ruang untuk peningkatan cakupan, Precision yang tinggi memastikan bahwa rekomendasi yang disajikan kepada pengguna adalah yang paling relevan.
