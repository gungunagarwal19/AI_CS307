from collections import deque

class State:
    def __init__(self, left_m, left_c, right_m, right_c, boat):
        self.left_m = left_m
        self.left_c = left_c
        self.right_m = right_m
        self.right_c = right_c
        self.boat = boat

    def is_valid(self):
        if self.left_m < 0 or self.left_c < 0 or self.right_m < 0 or self.right_c < 0:
            return False
        if (self.left_m > 0 and self.left_m < self.left_c) or (self.right_m > 0 and self.right_m < self.right_c):
            return False
        return True

    def is_goal(self):
        return self.left_m == 0 and self.left_c == 0

    def __hash__(self):
        return hash((self.left_m, self.left_c, self.right_m, self.right_c, self.boat))

    def __eq__(self, other):
        return (self.left_m, self.left_c, self.right_m, self.right_c, self.boat) == (other.left_m, other.left_c, other.right_m, other.right_c, other.boat)

    def __str__(self):
        return f'Left: M:{self.left_m}, C:{self.left_c} | Right: M:{self.right_m}, C:{self.right_c} | Boat: {"Left" if self.boat == 1 else "Right"}'

def get_successors(state):
    moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
    successors = []
    for m, c in moves:
        if state.boat == 1:
            new_state = State(state.left_m - m, state.left_c - c, state.right_m + m, state.right_c + c, 0)
        else:
            new_state = State(state.left_m + m, state.left_c + c, state.right_m - m, state.right_c - c, 1)
        if new_state.is_valid():
            successors.append(new_state)
    return successors

def bfs():
    start = State(3, 3, 0, 0, 1)
    queue = deque([(start, [])])
    explored = set()
    while queue:
        current_state, path = queue.popleft()
        if current_state.is_goal():
            return path + [current_state]
        explored.add(current_state)
        for successor in get_successors(current_state):
            if successor not in explored and successor not in (s[0] for s in queue):
                queue.append((successor, path + [current_state]))
    return None

def dfs():
    start = State(3, 3, 0, 0, 1)
    stack = [(start, [])]
    explored = set()
    while stack:
        current_state, path = stack.pop()
        if current_state.is_goal():
            return path + [current_state]
        explored.add(current_state)
        for successor in get_successors(current_state):
            if successor not in explored:
                stack.append((successor, path + [current_state]))
    return None

def print_solution(solution):
    if solution:
        print("Solution found:")
        for step in solution:
            print(step)
    else:
        print("No solution found.")

print("BFS Solution:")
bfs_solution = bfs()
print_solution(bfs_solution)

print("\nDFS Solution:")
dfs_solution = dfs()
print_solution(dfs_solution)
