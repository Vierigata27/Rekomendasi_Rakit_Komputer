from flask import Flask, jsonify
import os
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

class RekomendasiRakitan:
    def __init__(self, komponen_df, pop_size=100, generations=500, budget=10000000):
        self.komponen_df = komponen_df
        self.pop_size = pop_size
        self.generations = generations
        self.budget = budget
        self.history = []

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

    def random_component(self, category_id):
        available_components = self.komponen_df[self.komponen_df['id_kategori'] == category_id]
        return available_components.sample().iloc[0].to_dict() if not available_components.empty else None

    def calculate_fitness(self, individual):
        total_harga = sum(comp['harga_komponen'] for comp in individual.values() if comp)
        total_performa_minmax = sum(comp['performa_komponen'] for comp in individual.values() if comp)
        total_performa_asli = sum(comp['performa_asli'] for comp in individual.values() if comp)
        compatibility = 1 if self.check_compatibility(individual) else 0

        if total_harga > self.budget:
            return 0
        
        return (total_performa_minmax * compatibility) / ((self.budget - total_harga) + 1)

    def check_compatibility(self, individual):
        cpu = individual.get('CPU')
        motherboard = individual.get('Motherboard')
        psu = individual.get('Power Supply')
        gpu = individual.get('GPU')

        if not cpu or not motherboard or not psu or not gpu:
            return False

        soket_compatible = cpu['soket_komponen'] == motherboard['soket_komponen']
        daya_cukup = psu['daya_komponen'] > (cpu['daya_komponen'] + gpu['daya_komponen'])

        return soket_compatible and daya_cukup

    def crossover(self, parent1, parent2, crossover_rate=0.4):
        child = {}
        for key in parent1:
            if random.random() < crossover_rate:
                child[key] = parent2[key]
            else:
                child[key] = parent1[key]
        return child

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

            categories = list(category_map.keys())
            probabilities = [0.30 if cat in ["CPU", "GPU"] else 0.10 for cat in categories]
            category = random.choices(categories, probabilities)[0]

            new_component = self.random_component(category_map[category])
            if new_component:
                individual[category] = new_component

        return individual

    def run_genetic_algorithm(self):
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_overall_fitness = 0
        best_generation = 0
        best_overall_individual = None

        for generation in range(self.generations):
            population = sorted(population, key=lambda ind: self.calculate_fitness(ind), reverse=True)
            best_individual = population[0]
            best_fitness = self.calculate_fitness(best_individual)
            total_harga = sum(comp['harga_komponen'] for comp in best_individual.values() if comp)
            total_performa = sum(comp['performa_komponen'] for comp in best_individual.values() if comp)
            
            self.history.append({
                "Generasi": generation + 1,
                "Fitness": best_fitness,
                "Harga": total_harga,
                "Performa": total_performa
            })

            if best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                best_generation = generation + 1
                best_overall_individual = best_individual

            population = population[:self.pop_size // 2]
            new_population = [best_individual]
            for _ in range(self.pop_size - 1):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2, crossover_rate=0.4)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        return best_overall_individual, best_overall_fitness, best_generation, self.history

@app.route('/')
def index():
    return jsonify({"Terkoneksi": "Welcome to your Flask app 🚅"})

    @app.route('/rekomendasi', methods=['POST'])
    def get_rekomendasi():
        if request.content_type != 'application/json':
            return jsonify({"error": "Content-Type must be application/json"}), 415

        data = request.get_json()
        if not data or 'budget' not in data or 'komponen' not in data:
            return jsonify({"error": "Invalid JSON data"}), 400

        try:
            budget = int(data['budget'])
            komponen_data = data['komponen']

            komponen_df = pd.DataFrame(komponen_data)

            # Simpan performa asli sebelum normalisasi
            komponen_df['performa_asli'] = komponen_df['performa_komponen'].copy()

            # Normalisasi
            scaler = MinMaxScaler()
            komponen_df['performa_komponen'] = scaler.fit_transform(komponen_df[['performa_komponen']])

            rekomendasi = RekomendasiRakitan(komponen_df, budget=budget)
            best_rakitan, best_fitness, best_generation, history = rekomendasi.run_genetic_algorithm()

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


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
