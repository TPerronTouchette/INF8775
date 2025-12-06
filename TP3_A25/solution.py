import time
import math
import random
from typing import List, Tuple, Dict


class SpongeGridProblem:
    """
    Représente une instance du problème.

    - n: dimension du carré n×n à produire
    - alphabet: liste de symboles permis (ex: ["A","B","C","I"])
    - targets: séquences qu'on veut VOIR dans la grille
    - banned: séquences qu'on veut ÉVITER dans la grille
    """
    def __init__(
        self,
        n: int,
        alphabet: List[str],
        targets: List[str],
        banned: List[str]
    ):
        self.n = n
        self.alphabet = alphabet
        self.targets = targets
        self.banned = banned

def sequences_in_line(line: str, seq: str) -> bool:
    """True si 'seq' apparaît comme sous-chaîne contiguë dans 'line'."""
    return seq in line


def count_matches_in_grid(grid: List[List[str]], seqs: List[str]) -> int:
    """
    Combien de séquences DISTINCTES dans `seqs` apparaissent au moins une fois
    dans le carré (en ligne gauche->droite ou en colonne haut->bas) ?
    """
    n = len(grid)
    # Prépare toutes les lignes (strings)
    rows = ["".join(grid[i][j] for j in range(n)) for i in range(n)]
    cols = ["".join(grid[i][j] for i in range(n)) for j in range(n)]

    found = 0
    for s in seqs:
        present = any(sequences_in_line(r, s) for r in rows) \
                  or any(sequences_in_line(c, s) for c in cols)
        if present:
            found += 1
    return found


def score_grid(
    grid: List[List[str]],
    problem: SpongeGridProblem,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Dict[str, float]:
    """
    Calcule le score = alpha * (#objectifs couverts) - beta * (#bannies déclenchées)
    et renvoie aussi les sous-métriques.
    """
    covered_obj = count_matches_in_grid(grid, problem.targets) # O(n^2 * l * k) avec k = o donc O(n^2 * l * o)
    triggered_bad = count_matches_in_grid(grid, problem.banned) # O(n^2 * l * m) avec m = b donc O(n^2 * l * b)
    score_val = alpha * covered_obj - beta * triggered_bad
    return {
        "score": score_val,
        "covered": covered_obj,
        "triggered": triggered_bad
    }


# En partie insipiré par https://www.geeksforgeeks.org/dsa/implement-simulated-annealing-in-python/
# et https://fr.wikipedia.org/wiki/Recuit_simul%C3%A9

## Définition des variables

# Soit n la taille du carré (e.g. n=4 veut dire un carré de 4 x 4)

# Soit a la taille de l'alphabet

# Soit o le nombre d'objectifs

# Soit b le nombre de bannis 

def generate_weighted_grid(problem: SpongeGridProblem) -> List[List[str]]:
    """
    Generates an initial grid where characters are chosen based on their frequency in the target sequences.
    Total raw complexity: O(a + (o * l) + (a + o * l) + n^2)
    """
    counts = {char: 0 for char in problem.alphabet} # O(a) (voir définition des variables plus bas)
    total = 0

    for target in problem.targets: # O(o * l)
        for char in target: # O(l)
            counts[char] += 1
            total += 1

    weighted_chars = []

    for char, count in counts.items(): # O(a + o * l)
        weighted_chars.extend([char] * (count + 1))

    return [[random.choice(weighted_chars) for _ in range(problem.n)] for _ in range(problem.n)] # O(n^2)


def algo(problem: SpongeGridProblem, temperature: float = 15.0, cooling_rate: float = 0.9999, max_time_seconds: int = 180) -> List[List[str]]:
    """
    Simulated annealing algorithm to optimize the grid configuration for the given problem.
    Args:
        problem: An instance of SpongeGridProblem.
        temperature: Initial temperature for the annealing process.
        cooling_rate: Rate at which the temperature decreases.
        max_time_seconds: Maximum time to run the algorithm.
    Returns:
        A 2D list representing the optimized grid the algorithm was able to find.
    """
    start_time = time.time()
    
    MIN_TEMPERATURE = 0.001
    MAX_STAGNATION = 10_000
    
    current_grid = generate_weighted_grid(problem) # O(a + (o * l) + n^2)
    
    current_score = score_grid(current_grid, problem)['score'] # O((o + b) * l * n^2)
    
    best_grid = [row[:] for row in current_grid] # O(n^2)
    best_score = current_score
    
    iteration = 0
    nb_iterations_without_improvement = 0
    
    while (time.time() - start_time) < max_time_seconds:
        
        if temperature < MIN_TEMPERATURE:
            break

        if nb_iterations_without_improvement > MAX_STAGNATION:
            break

        random_x, random_y = random.randint(0, problem.n - 1), random.randint(0, problem.n - 1)
        old_character = current_grid[random_x][random_y]
        new_character = random.choice(problem.alphabet)
        
        if new_character == old_character:
            continue 
            
        current_grid[random_x][random_y] = new_character
        
        new_score = score_grid(current_grid, problem)['score'] # O((o + b) * l * n^2)
        
        delta = new_score - current_score
        
        accept = False
        
        if delta > 0:
            accept = True
        else:
            try:
                probability = math.exp(delta / temperature)
            except OverflowError:
                probability = 0
            if random.random() < probability:
                accept = True
        
        if accept:
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_grid = [row[:] for row in current_grid] # O(n^2)
                nb_iterations_without_improvement = 0
            else:
                nb_iterations_without_improvement += 1
        else:
            current_grid[random_x][random_y] = old_character
            nb_iterations_without_improvement += 1
        
        if iteration % 100 == 0:
            temperature *= cooling_rate
            
        iteration += 1
            
    return best_grid

if __name__ == '__main__':
    # Exemple d'utilisation

    banned = ["AB", "BA"]
    targets = ["AA", "BB"]
    alphabet = ["A", "B"]
    n = 5

    problem = SpongeGridProblem(n=n, alphabet=alphabet, targets=targets, banned=banned)
    algo(problem)