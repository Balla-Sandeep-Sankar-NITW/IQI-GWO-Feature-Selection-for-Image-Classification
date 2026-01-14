
# IQI-GWO ALGORITHM (with explicit validation)

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class IQI_GWO:
    def __init__(self, n_features, population_size=20, max_iters=30, quantum_rate=0.3,
                 mutation_rate=0.1, penalty_coef=0.01):
        self.n_features = n_features
        self.population_size = population_size
        self.max_iters = max_iters
        self.quantum_rate = quantum_rate
        self.mutation_rate = mutation_rate
        self.penalty_coef = penalty_coef
        self.population = np.random.randint(0, 2, (population_size, n_features))
        for i in range(population_size):
            if self.population[i].sum() == 0:
                self.population[i][np.random.randint(0, n_features)] = 1
        self.alpha_pos = self.population[0].copy()
        self.beta_pos = self.population[1].copy()
        self.delta_pos = self.population[2].copy()
        self.alpha_score = self.beta_score = self.delta_score = -np.inf
        self.history = {"fitness": [], "accuracy": [], "features": []}

    def fitness_function(self, mask, X_train, y_train, X_val, y_val, clf):
        """Evaluate fitness using separate validation set."""
        if mask.sum() == 0:
            return -1, 0, 0
        X_train_sel = X_train[:, mask == 1]
        X_val_sel = X_val[:, mask == 1]

        try:
            clf.fit(X_train_sel, y_train)
            preds = clf.predict(X_val_sel)
            acc = accuracy_score(y_val, preds)
        except:
            acc = 0

        # Penalize feature count
        return acc - self.penalty_coef * (mask.sum() / self.n_features), acc, mask.sum()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def quantum_update(self, wolf):
        theta = np.random.uniform(0, 2 * np.pi, self.n_features)
        q = np.cos(theta) * wolf + np.sin(theta) * (1 - wolf)
        return (q > 0.5).astype(int)

    def mutate(self, wolf):
        for i in range(self.n_features):
            if np.random.rand() < self.mutation_rate:
                wolf[i] = 1 - wolf[i]
        if wolf.sum() == 0:
            wolf[np.random.randint(0, self.n_features)] = 1
        return wolf

    def optimize(self, X_train, y_train, X_val, y_val, clf):
        print(f"\n🐺 Running IQI-GWO with {self.population_size} wolves × {self.max_iters} iters")
        for it in tqdm(range(self.max_iters), desc="IQI-GWO Optimization"):
            for i in range(self.population_size):
                fit, acc, n = self.fitness_function(self.population[i], X_train, y_train, X_val, y_val, clf)
                if fit > self.alpha_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy()
                    self.alpha_score, self.alpha_pos = fit, self.population[i].copy()
                elif fit > self.beta_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = fit, self.population[i].copy()
                elif fit > self.delta_score:
                    self.delta_score, self.delta_pos = fit, self.population[i].copy()

            a = 2 - 2 * ((it / self.max_iters) ** 2)
            for i in range(self.population_size):
                if np.random.rand() < self.quantum_rate:
                    self.population[i] = self.quantum_update(self.population[i])
                else:
                    A1, C1 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
                    A2, C2 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
                    A3, C3 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
                    D_alpha = abs(C1 * self.alpha_pos - self.population[i])
                    D_beta = abs(C2 * self.beta_pos - self.population[i])
                    D_delta = abs(C3 * self.delta_pos - self.population[i])
                    X1 = self.alpha_pos - A1 * D_alpha
                    X2 = self.beta_pos - A2 * D_beta
                    X3 = self.delta_pos - A3 * D_delta
                    new = (X1 + X2 + X3) / 3
                    self.population[i] = (self.sigmoid(new) > 0.5).astype(int)
                if np.random.rand() < self.mutation_rate:
                    self.population[i] = self.mutate(self.population[i])

            fit, acc, n = self.fitness_function(self.alpha_pos, X_train, y_train, X_val, y_val, clf)
            self.history["fitness"].append(fit)
            self.history["accuracy"].append(acc)
            self.history["features"].append(n)

        print(f"✅ Optimization done | Best Fitness: {self.alpha_score:.4f}")
        return self.alpha_pos, self.history