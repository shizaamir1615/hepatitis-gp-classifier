import numpy as np
import pandas as pd
import random
import math
import copy
import statistics

# ---------------------- UTILITY FUNCTIONS ----------------------

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def div(x, y): return x if abs(y) < 0.001 else x / y

# ---------------------- GP CLASS ----------------------

class GeneticProgramming:
    def __init__(self, X, y, function_set, terminal_set, pop_size=100, max_depth=4,
                 max_generations=50, tournament_size=3, crossover_prob=0.9,
                 mutation_prob=0.1, elitism=True, early_stopping=5, isba=False):
        self.X = X
        self.y = y
        self.function_set = function_set
        self.terminal_set = terminal_set
        self.pop_size = pop_size
        self.max_depth = max_depth
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism = elitism
        self.early_stopping = early_stopping
        self.isba = isba
        self.population = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.fitness_cache = {}
        self.gen_best_fitness = []
        self.gen_avg_fitness = []

    def ramped_half_and_half(self):
        population = []
        depth_range = range(2, self.max_depth + 1)
        for _ in range(self.pop_size):
            depth = random.choice(depth_range)
            tree = self.generate_tree_full(depth) if random.random() < 0.5 else self.generate_tree_grow(depth)
            population.append(tree)
        return population

    def generate_tree_full(self, max_depth, current_depth=0):
        if current_depth == max_depth:
            terminal = random.choice(self.terminal_set)
            return {'type': 'terminal', 'value': terminal() if callable(terminal) else terminal}
        func = random.choice(self.function_set)
        args = [self.generate_tree_full(max_depth, current_depth + 1) for _ in range(2)]
        return {'type': 'function', 'function': func, 'args': args}

    def generate_tree_grow(self, max_depth, current_depth=0):
        if current_depth == max_depth or (random.random() < 0.5 and current_depth > 0):
            terminal = random.choice(self.terminal_set)
            return {'type': 'terminal', 'value': terminal() if callable(terminal) else terminal}
        func = random.choice(self.function_set)
        args = [self.generate_tree_grow(max_depth, current_depth + 1) for _ in range(2)]
        return {'type': 'function', 'function': func, 'args': args}

    def evaluate_tree(self, tree, x):
        if tree['type'] == 'terminal':
            return x[int(tree['value'][1:])] if isinstance(tree['value'], str) else tree['value']
        args = [self.evaluate_tree(arg, x) for arg in tree['args']]
        try:
            result = tree['function'](*args)
            return max(min(result, 1), -1)
        except:
            return 0

    def tree_hash(self, tree):
        if tree['type'] == 'terminal':
            return str(tree['value'])
        func_name = tree['function'].__name__
        args_hash = [self.tree_hash(arg) for arg in tree['args']]
        return f"{func_name}({','.join(args_hash)})"

    def f1_score(self, y_true, y_pred):
        tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    def fitness(self, tree):
        tree_hash = self.tree_hash(tree)
        if tree_hash in self.fitness_cache:
            return self.fitness_cache[tree_hash]
        try:
            pred = [1 if self.evaluate_tree(tree, x) >= 0.5 else 0 for x in self.X]
            f1 = self.f1_score(self.y, pred)
            fitness = 1 - f1
            self.fitness_cache[tree_hash] = fitness
            return fitness
        except:
            self.fitness_cache[tree_hash] = float('inf')
            return float('inf')

    def tournament_selection(self):
        selected = random.sample(range(len(self.population)), self.tournament_size)
        best_idx = min(selected, key=lambda i: self.fitness(self.population[i]))
        return self.population[best_idx]

    def select_subtree(self, tree):
        if tree['type'] == 'terminal' or random.random() < 0.1:
            return [], tree
        arg_idx = random.randrange(len(tree['args']))
        path, subtree = self.select_subtree(tree['args'][arg_idx])
        return [arg_idx] + path, subtree

    def replace_subtree(self, tree, path, new_subtree):
        if not path:
            return copy.deepcopy(new_subtree)
        result = copy.deepcopy(tree)
        current = result
        for idx in path[:-1]:
            current = current['args'][idx]
        current['args'][path[-1]] = copy.deepcopy(new_subtree)
        return result

    def crossover(self, parent1, parent2):
        path1, subtree1 = self.select_subtree(parent1)
        path2, subtree2 = self.select_subtree(parent2)
        return self.replace_subtree(parent1, path1, subtree2), self.replace_subtree(parent2, path2, subtree1)

    def mutation(self, tree):
        path, _ = self.select_subtree(tree)
        depth = random.randint(1, 3)
        new_subtree = self.generate_tree_full(depth) if random.random() < 0.5 else self.generate_tree_grow(depth)
        return self.replace_subtree(tree, path, new_subtree)

    def prune_tree(self, tree, max_depth, current_depth=0):
        if current_depth >= max_depth:
            terminal = random.choice(self.terminal_set)
            return {'type': 'terminal', 'value': terminal() if callable(terminal) else terminal}
        if tree['type'] == 'terminal':
            return tree
        pruned_tree = copy.deepcopy(tree)
        pruned_tree['args'] = [self.prune_tree(arg, max_depth, current_depth + 1) for arg in pruned_tree['args']]
        return pruned_tree

    

    def gsim_index(self, tree1, tree2):
        """Global similarity index: Cosine similarity of tree evaluations."""
        try:
            pred1 = np.array([self.evaluate_tree(tree1, x) for x in self.X])
            pred2 = np.array([self.evaluate_tree(tree2, x) for x in self.X])
            norm1 = np.linalg.norm(pred1)
            norm2 = np.linalg.norm(pred2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return np.dot(pred1, pred2) / (norm1 * norm2)
        except:
            return 0

    def lsim_index(self, tree1, tree2):
        """Local similarity index: Structural overlap of trees."""
        def node_count(tree):
            if tree['type'] == 'terminal':
                return [(tree['value'], 1)]
            counts = []
            counts.append((tree['function'].__name__, 1))
            for arg in tree['args']:
                for node, count in node_count(arg):
                    counts.append((node, count))
            return counts
        try:
            nodes1 = dict(node_count(tree1))
            nodes2 = dict(node_count(tree2))
            common = sum(min(nodes1.get(k, 0), nodes2.get(k, 0)) for k in set(nodes1) | set(nodes2))
            total = sum(nodes1.values()) + sum(nodes2.values()) - common
            return common / total if total > 0 else 0
        except:
            return 0

    def isba_run(self, m=5, n=3, d=2):
        """ISBA: m global runs, n local runs per global, fix subtrees of depth d."""
        best_global_individual = None
        best_global_fitness = float('inf')
        for _ in range(m):  # Global runs
            self.population = self.ramped_half_and_half()
            for gen in range(self.max_generations):
                fitness_values = [self.fitness(ind) for ind in self.population]
                best_idx = np.argmin(fitness_values)
                if fitness_values[best_idx] < best_global_fitness:
                    best_global_fitness = fitness_values[best_idx]
                    best_global_individual = copy.deepcopy(self.population[best_idx])
                new_population = []
                if self.elitism:
                    new_population.append(copy.deepcopy(self.population[best_idx]))
                while len(new_population) < self.pop_size:
                    if random.random() < self.crossover_prob:
                        p1, p2 = self.tournament_selection(), self.tournament_selection()
                        c1, c2 = self.crossover(p1, p2)
                        c1 = self.mutation(c1) if random.random() < self.mutation_prob else c1
                        c2 = self.mutation(c2) if random.random() < self.mutation_prob else c2
                        new_population.extend([self.prune_tree(c1, self.max_depth), self.prune_tree(c2, self.max_depth)])
                    else:
                        parent = self.tournament_selection()
                        child = self.mutation(parent)
                        new_population.append(self.prune_tree(child, self.max_depth))
                self.population = new_population[:self.pop_size]
            
            local_population = [copy.deepcopy(best_global_individual) for _ in range(self.pop_size)]
            for _ in range(n):
                
                for i in range(len(local_population)):
                    path, subtree = self.select_subtree(local_population[i])
                    if len(path) > 0:
                        new_subtree = self.generate_tree_grow(d) if random.random() < 0.5 else self.generate_tree_full(d)
                        local_population[i] = self.replace_subtree(local_population[i], path, new_subtree)
                
                fitness_values = [self.fitness(ind) for ind in local_population]
                best_local_idx = np.argmin(fitness_values)
                if fitness_values[best_local_idx] < best_global_fitness:
                    best_global_fitness = fitness_values[best_local_idx]
                    best_global_individual = copy.deepcopy(local_population[best_local_idx])
                new_local_pop = [copy.deepcopy(local_population[best_local_idx])]
                while len(new_local_pop) < self.pop_size:
                    p1, p2 = random.choice(local_population), random.choice(local_population)
                    if self.gsim_index(p1, p2) > 0.7:  
                        continue
                    c1, c2 = self.crossover(p1, p2)
                    c1 = self.mutation(c1) if random.random() < self.mutation_prob else c1
                    c2 = self.mutation(c2) if random.random() < self.mutation_prob else c2
                    new_local_pop.extend([self.prune_tree(c1, self.max_depth), self.prune_tree(c2, self.max_depth)])
                local_population = new_local_pop[:self.pop_size]
        self.best_individual = best_global_individual
        self.best_fitness = best_global_fitness
        return self.best_individual

    def train(self):
        if self.isba:
            return self.isba_run(m=5, n=3, d=2)
        self.population = self.ramped_half_and_half()
        generations_no_improvement = 0
        for generation in range(1, self.max_generations + 1):
            fitness_values = [self.fitness(ind) for ind in self.population]
            best_idx = np.argmin(fitness_values)
            generation_best_fitness = fitness_values[best_idx]
            generation_best_individual = self.population[best_idx]
            avg_fitness = np.mean([f for f in fitness_values if not math.isinf(f)])
            self.gen_best_fitness.append(generation_best_fitness)
            self.gen_avg_fitness.append(avg_fitness)
            if generation_best_fitness < self.best_fitness:
                self.best_fitness = generation_best_fitness
                self.best_individual = copy.deepcopy(generation_best_individual)
                generations_no_improvement = 0
            else:
                generations_no_improvement += 1
            if generations_no_improvement >= self.early_stopping:
                break
            new_population = []
            if self.elitism:
                new_population.append(copy.deepcopy(generation_best_individual))
            while len(new_population) < self.pop_size:
                if random.random() < self.crossover_prob:
                    p1, p2 = self.tournament_selection(), self.tournament_selection()
                    c1, c2 = self.crossover(p1, p2)
                    c1 = self.mutation(c1) if random.random() < self.mutation_prob else c1
                    c2 = self.mutation(c2) if random.random() < self.mutation_prob else c2
                    new_population.extend([self.prune_tree(c1, self.max_depth), self.prune_tree(c2, self.max_depth)])
                else:
                    parent = self.tournament_selection()
                    child = self.mutation(parent)
                    new_population.append(self.prune_tree(child, self.max_depth))
            self.population = new_population[:self.pop_size]
        return self.best_individual

    def predict(self, X_test):
        return np.array([(self.evaluate_tree(self.best_individual, x) if not math.isnan(self.evaluate_tree(self.best_individual, x)) else 0.5) for x in X_test])

# =================== SUPPORT FUNCTIONS ===================

def tree_to_expression(tree):
    if tree['type'] == 'terminal':
        return str(tree['value'])
    op = {add: '+', sub: '-', mul: '*', div: '/'}[tree['function']]
    return f"({tree_to_expression(tree['args'][0])} {op} {tree_to_expression(tree['args'][1])})"

def compute_metrics(y_true, y_pred):
    
    tp = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))  
    tn = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))  
    fp = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))  
    fn = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true))) 
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1, tp, tn, fp, fn

def run_gp_with_seed(run_number, total_runs, seed, X_train, y_train, X_test, y_test, params, isba=False):
    print(f"\nRun {run_number}/{total_runs}: Computing... (Seed {seed}, {'ISBA' if isba else 'Regular GP'})")
    random.seed(seed)
    np.random.seed(seed)
    params['isba'] = isba
    gp = GeneticProgramming(X_train, y_train, **params)
    best_individual = gp.train()

    # Train metrics
    train_pred = gp.predict(X_train)
    train_class_preds = (train_pred >= 0.5).astype(int)
    train_acc, train_prec, train_rec, train_f1, train_tp, train_tn, train_fp, train_fn = compute_metrics(y_train, train_class_preds)

    # Test metrics
    test_pred = gp.predict(X_test)
    test_class_preds = (test_pred >= 0.5).astype(int)
    test_acc, test_prec, test_rec, test_f1, test_tp, test_tn, test_fp, test_fn = compute_metrics(y_test, test_class_preds)

    # Full dataset
    full_X = np.vstack((X_train, X_test))
    full_y = np.hstack((y_train, y_test))
    full_pred = gp.predict(full_X)
    full_class_preds = (full_pred >= 0.5).astype(int)
    full_acc, full_prec, full_rec, full_f1, full_tp, full_tn, full_fp, full_fn = compute_metrics(full_y, full_class_preds)

    print(f"\n===== Seed {seed} Results ({'ISBA' if isba else 'Regular GP'}) =====")
    print(f"Train Accuracy:  {train_acc:.3f}")
    print(f"Train F1 Score:  {train_f1:.3f}")
    print(f"Test Accuracy:   {test_acc:.3f}")
    print(f"Test F1 Score:   {test_f1:.3f}")
    print(f"Test TP : {test_tp}, TN : {test_tn}, FP: {test_fp}, FN: {test_fn}")
    print(f"Full Dataset TP : {full_tp}, TN : {full_tn}, FP: {full_fp}, FN: {full_fn}")
    print(f"Best Fitness: {gp.best_fitness:.4f}")
    print(f"Expression: {tree_to_expression(best_individual)}")

    return {
        'seed': seed,
        'train_accuracy': train_acc,
        'train_f1': train_f1,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_tp': test_tp,
        'test_tn': test_tn,
        'test_fp': test_fp,
        'test_fn': test_fn,
        'full_tp': full_tp,
        'full_tn': full_tn,
        'full_fp': full_fp,
        'full_fn': full_fn,
        'best_fitness': gp.best_fitness,
        'expression': tree_to_expression(best_individual),
        'algo': 'ISBA' if isba else 'Regular GP'
    }


if __name__ == "__main__":
    
    data = pd.read_csv("hepatitis.tsv", sep='\t')
    data = data.drop_duplicates()
    data["target_binary"] = data["target"].map({2: 1, 1: 0})  
    # Oversampling to balance classes
    positive = data[data["target_binary"] == 1]  # 
    negative = data[data["target_binary"] == 0]  # 
    oversampled_negative = negative.sample(n=len(positive), replace=True, random_state=42)
    data_balanced = pd.concat([positive, oversampled_negative])
    X = data_balanced.drop(columns=["target", "target_binary"]).values
    y = data_balanced["target_binary"].values

    
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # GP configuration
    function_set = [add, sub, mul, div]
    terminal_set = [f"X{i}" for i in range(X.shape[1])] + [lambda: random.uniform(-5, 5)]

    params = {
        'function_set': function_set,
        'terminal_set': terminal_set,
        'pop_size': 200,
        'max_depth': 4,
        'max_generations': 50,
        'tournament_size': 3,
        'crossover_prob': 0.7,
        'mutation_prob': 0.3,
        'early_stopping': 10,
        'elitism': True
    }

    seeds = [42, 123, 234, 345, 686, 57, 68, 898, 349, 93]
    results_regular = []
    results_isba = []
    for i, seed in enumerate(seeds, 1):
        results_regular.append(run_gp_with_seed(i, len(seeds), seed, X_train, y_train, X_test, y_test, params, isba=False))
    for i, seed in enumerate(seeds, 1):
        results_isba.append(run_gp_with_seed(i, len(seeds), seed, X_train, y_train, X_test, y_test, params, isba=True))


    
    def avg(key, results): return sum(r[key] for r in results) / len(results)
    def std(key, results): return statistics.stdev([r[key] for r in results])

    print("\n===== Aggregated Results (Regular GP) =====")
    print(f"Average Train F1:  {avg('train_f1', results_regular):.3f}")
    print(f"Average Test F1:   {avg('test_f1', results_regular):.3f}")
    print(f"Std Dev Test F1:   {std('test_f1', results_regular):.3f}")
    print(f"Average Test TP: {avg('test_tp', results_regular):.2f}, TN : {avg('test_tn', results_regular):.2f}, FP: {avg('test_fp', results_regular):.2f}, FN: {avg('test_fn', results_regular):.2f}")
    print(f"Average Full TP: {avg('full_tp', results_regular):.2f}, TN : {avg('full_tn', results_regular):.2f}")

    print("\n===== Aggregated Results (ISBA) =====")
    print(f"Average Train F1:  {avg('train_f1', results_isba):.3f}")
    print(f"Average Test F1:   {avg('test_f1', results_isba):.3f}")
    print(f"Std Dev Test F1:   {std('test_f1', results_isba):.3f}")
    print(f"Average Test TP: {avg('test_tp', results_isba):.2f}, TN : {avg('test_tn', results_isba):.2f}, FP: {avg('test_fp', results_isba):.2f}, FN: {avg('test_fn', results_isba):.2f}")
    print(f"Average Full TP: {avg('full_tp', results_isba):.2f}, TN : {avg('full_tn', results_isba):.2f}")

    best_regular_idx = max(range(len(results_regular)), key=lambda i: results_regular[i]['test_f1'])
    best_isba_idx = max(range(len(results_isba)), key=lambda i: results_isba[i]['test_f1'])
    print(f"\nBest Regular GP Model (seed {results_regular[best_regular_idx]['seed']}):")
    print(f"Expression: {results_regular[best_regular_idx]['expression']}")
    print(f"Test F1 Score: {results_regular[best_regular_idx]['test_f1']:.3f}")
    print(f"\nBest ISBA Model (seed {results_isba[best_isba_idx]['seed']}):")
    print(f"Expression: {results_isba[best_isba_idx]['expression']}")
    print(f"Test F1 Score: {results_isba[best_isba_idx]['test_f1']:.3f}")
    