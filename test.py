from collections import deque

# Directions : H, B, G, D (haut, bas, gauche, droite)
MOVES = {
    'H': -3,  # Haut
    'B': 3,   # Bas
    'G': -1,  # Gauche
    'D': 1    # Droite
}

def is_valid_move(pos, move):
    if move == 'H': return pos >= 3
    if move == 'B': return pos <= 5
    if move == 'G': return pos % 3 != 0
    if move == 'D': return pos % 3 != 2
    return False

def apply_move(state, move):
    state = list(state)
    i = state.index(0)
    j = i + MOVES[move]
    state[i], state[j] = state[j], state[i]
    return tuple(state)

def bfs(start, goal):
    visited = set()
    queue = deque()
    queue.append((start, []))
    visited.add(start)

    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path

        for move in ['H', 'B', 'G', 'D']:  # ordre HBGD
            if is_valid_move(current.index(0), move):
                new_state = apply_move(current, move)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [move]))

    return None  # Pas de solution

# Exemple : état initial et but
etat_initial = (1, 0, 3, 4, 2, 6, 7, 5, 8)
etat_final = (1, 2, 3, 4, 5, 6, 7, 8, 0)

chemin = bfs(etat_initial, etat_final)
print("Chemin trouvé :", chemin)
print("Nombre de mouvements :", len(chemin) if chemin else "Aucune solution")
