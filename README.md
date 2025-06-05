# Laporan Proyek Machine Learning - TRAVU (CC25-CF163)

## Project Overview

Pada bagian ini, Kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)


Sektor pariwisata merupakan salah satu pilar ekonomi penting bagi Indonesia, berkontribusi signifikan terhadap Produk Domestik Bruto (PDB) dan menciptakan lapangan kerja. Dengan kekayaan alam, budaya, dan sejarah yang melimpah, Indonesia menawarkan beragam destinasi wisata yang menarik bagi wisatawan domestik maupun mancanegara. Namun, di tengah gemerlapnya pilihan, wisatawan seringkali kesulitan dalam menemukan tempat wisata yang sesuai dengan preferensi pribadi mereka. Informasi yang tersebar dan tidak terstruktur membuat proses pencarian menjadi tidak efisien, mengakibatkan potensi destinasi yang menarik terlewatkan.

Fenomena ini diperkuat oleh riset yang menunjukkan bahwa rekomendasi personal yang relevan dapat meningkatkan kepuasan dan keterlibatan pengguna dalam memilih destinasi wisata [1]. Tantangan utama terletak pada bagaimana sistem dapat memahami preferensi kompleks pengguna dan menyajikannya dalam bentuk rekomendasi yang akurat dan bermanfaat. Mengingat sebagian besar informasi wisata tersedia dalam bentuk teks deskriptif, teknik pemrosesan bahasa alami (NLP) menjadi krusial untuk mengekstrak makna dan kesamaan antar destinasi.

Oleh karena itu, proyek TRAVU ini bertujuan untuk mengembangkan sistem rekomendasi tempat wisata di Indonesia yang dipersonalisasi. Sistem ini akan memanfaatkan teknik Machine Learning, khususnya kombinasi dari pendekatan berbasis konten (Content-Based Filtering) dengan model neural network, untuk memberikan rekomendasi yang lebih cerdas dan relevan. Dengan menggabungkan informasi deskriptif wisata dan preferensi pengguna, diharapkan sistem ini dapat membantu wisatawan menemukan destinasi impian mereka dengan lebih mudah, sekaligus mendukung peningkatan sektor pariwisata di Indonesia.


## Business Understanding

Pada bagian ini, Anda perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Approach” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution approach (algoritma atau pendekatan sistem rekomendasi).
 

### Problem Statements

1. Kurangnya Personalisasi Rekomendasi: Wisatawan seringkali menghadapi kesulitan dalam menemukan destinasi yang benar-benar sesuai dengan minat dan preferensi spesifik mereka. Sistem rekomendasi yang ada mungkin hanya memberikan daftar umum atau populer, tanpa mempertimbangkan karakteristik unik yang dicari pengguna.
2. Overload Informasi: Dengan banyaknya pilihan destinasi wisata di Indonesia, wisatawan seringkali overwhelmed dengan informasi yang tersedia. Mereka membutuhkan cara yang efisien untuk menyaring dan mengidentifikasi tempat-tempat yang paling relevan dari berbagai kategori (misalnya, gunung, pantai, museum) dan lokasi.
3. Inefisiensi Pencarian: Proses pencarian manual tempat wisata membutuhkan waktu dan usaha yang signifikan. Wisatawan harus mencari informasi dari berbagai sumber, membandingkan fitur, dan membaca ulasan untuk membuat keputusan, yang seringkali memakan waktu dan melelahkan.

### Goals

1. Meningkatkan Relevansi Rekomendasi: Menyediakan rekomendasi tempat wisata yang sangat relevan dengan preferensi kategori pengguna (misalnya, "gunung", "taman", "pantai") di provinsi tertentu.
2. Mempermudah Pemilihan Destinasi: Mengurangi beban informasi bagi wisatawan dengan menyajikan daftar rekomendasi teratas yang sudah difilter dan diurutkan berdasarkan tingkat relevansi.
3. Optimalisasi Proses Pencarian: Mengotomatiskan dan mempercepat proses penemuan destinasi wisata melalui sistem rekomendasi berbasis Machine Learning, sehingga wisatawan dapat membuat keputusan dengan lebih cepat dan efisien.

### Solution Approach

#### Solution Statement 1: Content-Based Filtering dengan TF-IDF Cosine Similarity

Pendekatan:
Pendekatan ini berfokus pada konten atau deskripsi dari item (dalam hal ini, tempat wisata). Metode TF-IDF (Term Frequency-Inverse Document Frequency) akan digunakan untuk mengubah teks deskripsi tempat wisata dan preferensi kategori pengguna menjadi representasi numerik (vektor). Setelah itu, cosine similarity akan dihitung antara vektor preferensi pengguna dan vektor setiap tempat wisata. Skor cosine similarity yang lebih tinggi menunjukkan kemiripan yang lebih besar antara preferensi pengguna dan deskripsi wisata.

Kelebihan:

- Transparansi: Mudah dijelaskan mengapa suatu rekomendasi diberikan (karena kesamaan fitur).
- Tidak Membutuhkan Data Pengguna Lain: Sistem tidak memerlukan data dari pengguna lain; rekomendasi hanya didasarkan pada profil pengguna dan item itu sendiri, sehingga cocok untuk cold-start problem bagi pengguna baru.
- Fleksibel: Dapat beradaptasi dengan perubahan preferensi pengguna secara real-time jika profil preferensi diupdate.

Kekurangan:

- Over-specialization: Cenderung merekomendasikan item yang sangat mirip dengan yang sudah disukai pengguna, membatasi eksplorasi item baru.
- Membutuhkan Deskripsi Item yang Kaya: Kualitas rekomendasi sangat bergantung pada kekayaan dan kelengkapan deskripsi teks tempat wisata.
- Keterbatasan Pemahaman Konteks: Hanya melihat kesamaan kata kunci, mungkin melewatkan kesamaan konseptual atau kontekstual yang lebih dalam.

#### Solution Statement 2: Neural Network untuk Memprediksi Skor Relevansi

Pendekatan:
Pendekatan ini menggunakan model neural network (misalnya, Multi-Layer Perceptron) untuk memprediksi skor relevansi (probabilitas kesukaan) sebuah tempat wisata bagi pengguna. Fitur input untuk model akan mencakup informasi kategorikal yang telah di-encode (kategori wisata, provinsi), fitur numerik (panjang deskripsi), dan sebuah fitur biner yang menunjukkan apakah kategori wisata termasuk dalam preferensi pengguna. Model ini akan dilatih untuk memprediksi skor kesamaan yang dihasilkan dari cosine similarity sebagai target.

Kelebihan:

- Kemampuan Menangkap Pola Kompleks: Neural network mampu mempelajari hubungan non-linear dan pola kompleks antar fitur yang mungkin tidak dapat ditangkap oleh metode linier sederhana.
- Fleksibilitas Fitur: Dapat mengintegrasikan berbagai jenis fitur (kategorikal, numerik, biner) dengan mudah.
- Potensi untuk Peningkatan Kinerja: Jika dilatih dengan data yang cukup dan arsitektur yang tepat, model neural network dapat memberikan prediksi yang sangat akurat.

Kekurangan:

- Black Box: Sulit untuk menginterpretasikan mengapa model membuat rekomendasi tertentu (kurang transparan dibandingkan TF-IDF).
- Membutuhkan Data Pelatihan yang Cukup: Kinerja sangat bergantung pada kuantitas dan kualitas data pelatihan.
- Kompleksitas Komputasi: Pelatihan model neural network bisa membutuhkan sumber daya komputasi yang signifikan.

#### Pendekatan Kombinasi (Hybrid Approach):
Untuk mengatasi kekurangan masing-masing pendekatan dan memaksimalkan kelebihan, proyek ini akan mengadopsi pendekatan hibrida. Skor yang dihasilkan dari cosine similarity (TF-IDF) akan dijadikan ground truth atau target untuk melatih model neural network. Kemudian, dalam proses rekomendasi akhir, skor dari model neural network akan digabungkan dengan skor cosine similarity menggunakan bobot tertentu. Pendekatan ini memungkinkan model neural network untuk belajar dari kesamaan berbasis konten sambil tetap mempertahankan kemampuan untuk menangkap pola yang lebih kompleks dari fitur-fitur lain.


## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.


Dataset yang digunakan dalam proyek ini adalah wisata_indonesia_final.csv, yang dapat diakses melalui repositori GitHub Travu-Team. Dataset ini berisi informasi mengenai 1.025 objek wisata di seluruh Indonesia, dengan 11 kolom yang menyediakan detail lengkap untuk setiap destinasi.

Berikut adalah uraian variabel-variabel pada dataset wisata_indonesia_final.csv:

- kategori: Jenis atau kategori utama dari tempat wisata (misal: 'air terjun', 'pantai', 'museum', 'gunung').
- nama_wisata: Nama spesifik dari objek wisata. Setiap nama_wisata dalam dataset ini adalah unik, menunjukkan bahwa tidak ada duplikasi entri untuk tempat wisata yang sama.
- latitude: Koordinat lintang geografis lokasi wisata.
- longitude: Koordinat bujur geografis lokasi wisata.
- alamat: Alamat lengkap dari tempat wisata.
- provinsi: Provinsi tempat lokasi wisata berada. Terdapat 90 nilai kosong pada kolom ini yang perlu ditangani.
- kota_kabupaten: Kota atau kabupaten tempat lokasi wisata berada. Terdapat beberapa nilai kosong pada kolom ini.
- nama_lengkap: Nama lengkap atau nama alternatif dari tempat wisata, seringkali menyerupai nama_wisata.
- deskripsi: Deskripsi singkat atau informasi tentang tempat wisata. Kolom ini akan menjadi input utama untuk analisis teks (TF-IDF).
- path: Path relatif ke file data terkait (tidak digunakan dalam pemodelan).
- path_gambar: Path relatif ke file gambar terkait tempat wisata (tidak digunakan dalam pemodelan).

Dari analisis awal, dataset ini memiliki 1.025 entri unik dengan distribusi kategori dan provinsi yang bervariasi. Kolom provinsi dan kota_kabupaten memiliki nilai null, yang akan memerlukan penanganan khusus dalam tahapan data preparation.


## Exploratory Data Analysis and Visualization

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
