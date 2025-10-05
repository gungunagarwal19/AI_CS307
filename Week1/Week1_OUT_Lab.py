from collections import deque

# BFS for Rabbit Leap
def bfs_rabbit_leap(initial_state, goal_state):
    queue = deque([(initial_state, [])])
    visited = set([tuple(initial_state)])
    
    while queue:
        current_state, path = queue.popleft()

        if current_state == goal_state:
            return path + [current_state]

        for i in range(len(current_state)):
            if current_state[i] == '.':
                if i > 0 and current_state[i - 1] != '.':
                    new_state = current_state[:]
                    new_state[i], new_state[i - 1] = new_state[i - 1], new_state[i]
                    if tuple(new_state) not in visited:
                        visited.add(tuple(new_state))
                        queue.append((new_state, path + [current_state]))

                if i > 1 and current_state[i - 1] != '.' and current_state[i - 2] != '.':
                    new_state = current_state[:]
                    new_state[i], new_state[i - 2] = new_state[i - 2], new_state[i]
                    if tuple(new_state) not in visited:
                        visited.add(tuple(new_state))
                        queue.append((new_state, path + [current_state]))

                if i < len(current_state) - 1 and current_state[i + 1] != '.':
                    new_state = current_state[:]
                    new_state[i], new_state[i + 1] = new_state[i + 1], new_state[i]
                    if tuple(new_state) not in visited:
                        visited.add(tuple(new_state))
                        queue.append((new_state, path + [current_state]))

                if i < len(current_state) - 2 and current_state[i + 1] != '.' and current_state[i + 2] != '.':
                    new_state = current_state[:]
                    new_state[i], new_state[i + 2] = new_state[i + 2], new_state[i]
                    if tuple(new_state) not in visited:
                        visited.add(tuple(new_state))
                        queue.append((new_state, path + [current_state]))

    return None

# DFS for Rabbit Leap
def dfs_rabbit_leap(current_state, goal_state, visited):
    if current_state == goal_state:
        return [current_state]
    
    visited.add(tuple(current_state))
    
    for i in range(len(current_state)):
        if current_state[i] == '.':
            if i > 0 and current_state[i - 1] != '.':
                new_state = current_state[:]
                new_state[i], new_state[i - 1] = new_state[i - 1], new_state[i]
                if tuple(new_state) not in visited:
                    result = dfs_rabbit_leap(new_state, goal_state, visited)
                    if result is not None:
                        return [current_state] + result

            if i > 1 and current_state[i - 1] != '.' and current_state[i - 2] != '.':
                new_state = current_state[:]
                new_state[i], new_state[i - 2] = new_state[i - 2], new_state[i]
                if tuple(new_state) not in visited:
                    result = dfs_rabbit_leap(new_state, goal_state, visited)
                    if result is not None:
                        return [current_state] + result

            if i < len(current_state) - 1 and current_state[i + 1] != '.':
                new_state = current_state[:]
                new_state[i], new_state[i + 1] = new_state[i + 1], new_state[i]
                if tuple(new_state) not in visited:
                    result = dfs_rabbit_leap(new_state, goal_state, visited)
                    if result is not None:
                        return [current_state] + result

            if i < len(current_state) - 2 and current_state[i + 1] != '.' and current_state[i + 2] != '.':
                new_state = current_state[:]
                new_state[i], new_state[i + 2] = new_state[i + 2], new_state[i]
                if tuple(new_state) not in visited:
                    result = dfs_rabbit_leap(new_state, goal_state, visited)
                    if result is not None:
                        return [current_state] + result
    
    return None

def solve_rabbit_leap_dfs(initial_state, goal_state):
    visited = set()
    result = dfs_rabbit_leap(initial_state, goal_state, visited)
    if result:
        return result
    else:
        return "No solution found."

# Test BFS and DFS for Rabbit Leap
initial_state = ['E', 'E', 'E', '.', 'W', 'W', 'W']
goal_state = ['W', 'W', 'W', '.', 'E', 'E', 'E']

# BFS Solution
print("BFS Solution for Rabbit Leap:")
bfs_solution = bfs_rabbit_leap(initial_state, goal_state)
if bfs_solution:
    for step in bfs_solution:
        print(step)
else:
    print("No solution found.")

# DFS Solution
print("\nDFS Solution for Rabbit Leap:")
dfs_solution = solve_rabbit_leap_dfs(initial_state, goal_state)
if dfs_solution != "No solution found.":
    for state in dfs_solution:
        print(state)
else:
    print(dfs_solution)
