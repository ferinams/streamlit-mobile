# Laporan Proyek Machine Learning 
### Nama  : Ferina Melania Sari
### NIM   : 211351056
### Kelas : Informatika Malam A
Estimasi harga telepon genggam (Hp) ini bisa digunakan bagi semua orang yang ingin mengetahui harga Hp dengan cara mengecek RAM, memori internal, prosesor dan lain-lain secara online.

## Business Understanding
Bisa mengetahui spek Hp yang diinginkan tanpa harus pergi ke pusat pembelian hp terlebih dahulu.
Bagian laporan ini mencangkup:

### Problem Statemens
kebanyakan orang ingin tahu harga terlebih dahulu mau itu hanya untuk cek harga saja ataupun mencari tahu untuk menabung terlebih dahulu sebelum membeli Hp yang diinginkan

### Goals
Mencari solusi untuk memudahkan orang yang sedang mencari harga Hp sebelum membeli

### Solution Steatment
- Membuat Platform untuk mencari harga hp berbasis web, yang pertama adalah membuat sebuah platform pencarian harga hp berbasis web yang mengintegrasi data dari Kaggle.com untuk memudahkan orang yang sedang mencari informasi seputar harga hp. dalam Platform ini pengguna bisa tau seputar harga hp berdasarkan spek yang pegguna inginkan.
- Model yng dihasilkan dari dataset itu menggunakan linear Regresion.

### Data Understanding
Dataset yang saya gunakan berasal jadi Kaggle yang berisi harga telepon genggam (hp). Dataset ini merupakan sekumpulan data yang dikumpulkan dari website populer tentang harga hp terbaru. Dataset ini mengandung 1359 baris dan lebih dari 8 columns setelah dilakukan data cleaning.

##### Variabel-variabel pada platform estimasi harga telepon genggam adalah sebagai berikut :
Battery_capacity_mAhC = menginput kapasitas baterai (int64) <br>
Screen_size_inches = menginput input ukuran layar (float64) <br>
Processor = menginput Processor (int64) <br>
RAM_MB = menginput besarnya RAM (int64) <br>
Internal_storage_GB = menginput besarnya memori internal (float64) <br>
Rear_camera = menginput pixel kamera belakang (float64) <br>                       
Number_of_SIMs = menginput banyaknya sim card (int64) <br>
Resolution_x   = menginputkan resolusi layar (int64) <br>

### Data Preparation

#### Data Collection
Untuk data Collection ini,saya mendapatkan dataset yang nantinya digunakan dari website Kaggle dengan nama dataset Mobile Phone Specifications and Prices, jika anda tertarik dengan datasetnya, anda bisa klik link.


#### Data Discovery And Profiling
Untuk bagian ini kita gunakan teknik EDA
Pertama Kita mengimport semua library yang dibutuhkan
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

setelah itu kita panggil dataset
```python
df = pd.read_csv('mobile_phone.csv')
```

untuk melihat mengenai type data dari masing masing kolom kita bisa menggunakan properti info,
```python
df.info()
```

Selanjutnya kita akan memeriksa apakah datasetsnya terdapat baris yang kosong atau null dengan menggunakan seaborn,
```python
sns.heatmap(df.isnull())
```

kita lanjut dengan data exploration kita,
```python
numeric_df = df.select_dtypes(include=['int64', 'float64'])
```

```python
plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(),annot=True)
plt.title("heatmap korelasi (numerik)")
plt.show()
```

selanjutnya kita akan melihat grafik jenis prossesor berdasarkan model hp,

```python
models = df.groupby('Operating system').count()[['Processor']].sort_values(by='Processor',ascending=True).reset_index()
models = models.rename(columns={'Processor' : 'numberOfmobile'})
```

```python
pig = plt.figure(figsize=(15,5))
sns.barplot(x=models['Operating system'], y=models['numberOfmobile'], color='royalblue')
plt.xticks(rotation=60)
plt.show()
```

lanjut kita cek grafik distribusi kapasitas baterai,

```python
plt.figure(figsize=(15,5))
sns.displot(df['Battery capacity (mAh)'])
```
setelah itu cek grafik distribusi harganya,

```python
plt.figure(figsize=(15,5))
sns.displot(df['Price'])
```

### Modeling
Langkah pertama masukan kolom kolom fitur yang ada didatasets dab kolom targetnya'

```python
features = ['Processor', 'Screen size (inches)', 'Price', 'RAM (MB)', 'Battery capacity (mAh)', 'Rear camera', 'Number of SIMs', 'Resolution x']
x = df[features]
y = df['Rear camera']
x.shape, y.shape
```
Selanjutnya kita akan menentukan berapa persen dari datasets yang akan digunakan untuk test dan untuk train,

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=36)
y_test.shape
```

Mari kita lanjut dengan membuat model Linear Regressionnya,

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
```
lalu kita keluarkan hasil prediksi yang didapat,

```python
score = lr.score(x_test, y_test)
print('akurasi model regresi linier =', score)
```
hasil yang kita dapat 100% ternyata sangat sempurna, mari kita test menggunakan sebuah array value,

```python
input_data = np.array([[4015, 6.7, 8, 4, 120, 68, 2, 1660]])

prediction = lr.predict(input_data)
print('Estimasi harga produk dalam IDR :', prediction)
```
akhirnya modelnya sudah selesai, mari kita export sebagai sav agar nanti bisa kita gunakan pada project web streamlit kita.

```python
import pickle

filename = 'mobile_phone.sav'
pickle.dump(lr,open(filename,'wb'))
```

### Evalution

### Deployment
[klik disini] (https://app-mobile-fe2vghsuqdqckh7xm68rln.streamlit.app/)

![gambar](https://github.com/ferinams/streamlit-mobile/assets/149289420/019eb9a4-1a05-4079-8475-23bf672c8c53)






