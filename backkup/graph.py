import sqlite3
import matplotlib.pyplot as plt

# Membuat koneksi ke database SQLite
conn = sqlite3.connect('nama_database.db')
cursor = conn.cursor()

# Lakukan kueri SQL untuk mengekstrak data skor
sql_query = "SELECT score FROM jawaban"
cursor.execute(sql_query)

# Ambil semua hasil
results = cursor.fetchall()

# Menutup koneksi database setelah pengambilan data
conn.close()

# Mengubah hasil menjadi list skor
scores = [result[0] for result in results]

# Membuat histogram dari data skor
plt.hist(scores, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Distribution of Scores')
plt.grid(True)
plt.show()
