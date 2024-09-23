D = 50
N = 100

min_steps = N + 2
max_steps = N + 2*N
shortest_tour = int(D/9) + 1
N_shortest_torus = (int(N/shortest_tour)+1)
better_bound = (N_shortest_torus)*shortest_tour  + 2*N_shortest_torus

print("N_shortest_torus", N_shortest_torus)
print("shortest tour", shortest_tour)
print(min_steps, max_steps, better_bound)