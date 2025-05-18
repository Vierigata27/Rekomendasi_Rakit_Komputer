# Import library yang diperlukan
from flask import Flask, jsonify, request  # Untuk membuat API Flask
import os
import pandas as pd                       # Untuk mengelola data komponen
import random                             # Untuk operasi acak (mutasi, crossover, dll)
from sklearn.preprocessing import MinMaxScaler  # Untuk normalisasi performa

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Kelas utama untuk sistem rekomendasi berbasis algoritma genetika
class RekomendasiRakitan:
    def __init__(self, komponen_df, pop_size=100, generations=100, budget=10000000):
        self.komponen_df = komponen_df  # Dataset komponen
        self.pop_size = pop_size        # Ukuran populasi
        self.generations = generations  # Jumlah generasi
        self.budget = budget            # Batasan anggaran
        self.history = []               # Menyimpan histori evolusi

    # Membuat satu individu (rakitan komputer)
    def create_individual(self):
        return {
            'CPU': self.random_component(1),
            'Motherboard': self.random_component(2),
            'GPU': self.random_component(3),
            'RAM': self.random_component(4),
            'Storage': self.random_component(5),
            'Power Supply': self.random_component(6),
            'Casing': self.random_component(7),
            'Fan CPU': self.random_component(8),
        }

    # Memilih komponen secara acak berdasarkan ID kategori
    def random_component(self, category_id):
        available_components = self.komponen_df[self.komponen_df['id_kategori'] == category_id]
        return available_components.sample().iloc[0].to_dict() if not available_components.empty else None

    # Fungsi untuk menghitung nilai fitness dari rakitan
    def calculate_fitness(self, individual):
        total_harga = sum(comp['harga_komponen'] for comp in individual.values() if comp)
        total_performa_minmax = sum(comp['performa_komponen'] for comp in individual.values() if comp)
        total_performa_asli = sum(comp['performa_asli'] for comp in individual.values() if comp)
        compatibility = 1 if self.check_compatibility(individual) else 0

        # Jika harga melebihi anggaran, fitness-nya 0
        if total_harga > self.budget:
            return 0

        # Fitness didasarkan pada total performa * kompatibilitas
        return (total_performa_minmax * compatibility)

    # Mengecek apakah komponen-komponen kompatibel
    def check_compatibility(self, individual):
        cpu = individual.get('CPU')
        motherboard = individual.get('Motherboard')
        psu = individual.get('Power Supply')
        gpu = individual.get('GPU')

        if not cpu or not motherboard or not psu or not gpu:
            return False

        # Cek soket CPU dan motherboard
        soket_compatible = cpu['soket_komponen'] == motherboard['soket_komponen']

        # PSU harus memiliki daya minimal 40% lebih tinggi dari total CPU + GPU
        daya_cukup = psu['daya_komponen'] >= ((cpu['daya_komponen'] + gpu['daya_komponen']) * 1.4)

        # Rasio performa CPU dan GPU harus seimbang
        cpu_banchmark = cpu['performa_asli']
        gpu_banchmark = gpu['performa_asli']
        ratio = cpu_banchmark / gpu_banchmark
        rasio_seimbang = 0.3 <= ratio <= 5

        return soket_compatible and daya_cukup and rasio_seimbang

    # Fungsi crossover antara dua individu (persilangan)
    def crossover(self, parent1, parent2, crossover_rate=0.4):
        child = {}
        for key in parent1:
            # Komponen diambil dari parent1 atau parent2 berdasarkan probabilitas crossover
            child[key] = parent2[key] if random.random() < crossover_rate else parent1[key]
        return child

    # Fungsi mutasi (penggantian komponen secara acak)
    def mutate(self, individual, mutation_rate=0.4):
        if random.random() < mutation_rate:
            category_map = {
                "CPU": 1,
                "Motherboard": 2,
                "GPU": 3,
                "RAM": 4,
                "Storage": 5,
                "Power Supply": 6,
                "Casing": 7,
                "Fan CPU": 8,
            }

            # Komponen yang sering dimutasi: CPU & GPU (diberi probabilitas lebih besar)
            categories = list(category_map.keys())
            probabilities = [0.30 if cat in ["CPU", "GPU"] else 0.10 for cat in categories]
            category = random.choices(categories, probabilities)[0]

            # Pilih komponen baru secara acak dari kategori tersebut
            new_component = self.random_component(category_map[category])
            if new_component:
                individual[category] = new_component

        return individual

    # Fungsi utama untuk menjalankan algoritma genetika
    def run_genetic_algorithm(self):
        # Inisialisasi populasi awal
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_overall_fitness = 0
        best_generation = 0
        best_overall_individual = None

        # Proses evolusi
        for generation in range(self.generations):
            # Urutkan populasi berdasarkan fitness secara menurun
            population = sorted(population, key=lambda ind: self.calculate_fitness(ind), reverse=True)
            best_individual = population[0]
            best_fitness = self.calculate_fitness(best_individual)
            total_harga = sum(comp['harga_komponen'] for comp in best_individual.values() if comp)
            total_performa = sum(comp['performa_komponen'] for comp in best_individual.values() if comp)

            # Simpan histori tiap generasi
            self.history.append({
                "Generasi": generation + 1,
                "Fitness": best_fitness,
                "Harga": total_harga,
                "Performa": total_performa
            })

            # Update solusi terbaik jika ada peningkatan fitness
            if best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                best_generation = generation + 1
                best_overall_individual = best_individual

            # Seleksi elitisme: ambil separuh terbaik untuk jadi induk
            population = population[:self.pop_size // 2]

            # Buat populasi baru dari hasil crossover dan mutasi
            new_population = [best_individual]  # Selalu simpan individu terbaik
            for _ in range(self.pop_size - 1):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2, crossover_rate=0.4)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        return best_overall_individual, best_overall_fitness, best_generation, self.history

# Endpoint untuk memastikan API hidup
@app.route('/')
def index():
    return jsonify({"Terkoneksi": "Welcome to your Flask app"})

# Endpoint POST untuk menerima request rekomendasi rakitan
@app.route('/rekomendasi', methods=['POST'])
def get_rekomendasi():
    # Validasi content type
    if request.content_type != 'application/json':
        return jsonify({"error": "Content-Type must be application/json"}), 415

    # Validasi isi JSON
    data = request.get_json()
    if not data or 'budget' not in data or 'komponen' not in data:
        return jsonify({"error": "Invalid JSON data"}), 400

    try:
        # Ambil nilai budget dan data komponen
        budget = int(data['budget'])
        komponen_data = data['komponen']

        # Ubah ke dalam bentuk DataFrame
        komponen_df = pd.DataFrame(komponen_data)

        # Simpan nilai performa asli sebelum normalisasi
        komponen_df['performa_asli'] = komponen_df['performa_komponen'].copy()

        # Normalisasi performa komponen agar skala setara
        scaler = MinMaxScaler()
        komponen_df['performa_komponen'] = scaler.fit_transform(komponen_df[['performa_komponen']])

        # Jalankan sistem rekomendasi
        rekomendasi = RekomendasiRakitan(komponen_df, budget=budget)
        best_rakitan, best_fitness, best_generation, history = rekomendasi.run_genetic_algorithm()

        # Kembalikan hasil rekomendasi
        response = {
            "rakitan_terbaik": best_rakitan,
            "fitness_score": best_fitness,
            "generasi_terbaik": best_generation,
            "history": history
        }
        return jsonify(response)

    except ValueError:
        return jsonify({"error": "Budget harus berupa angka"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Menjalankan aplikasi Flask jika file ini dijalankan langsung
if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
