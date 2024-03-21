import matplotlib.pyplot as plt

# Define lists to store data
avg = []
games = []
max = []

# Read data from file
with open('train_data.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if "avg" in line:
            games.append(int(line.split()[0]))
            avg.append(float(line.split()[3]))
            max.append(float(line.split()[6]))

# Plotting
plt.plot(games, avg, label='Average')
plt.plot(games, max, label='Max')

plt.xlabel('Games')
plt.ylabel('Score')
plt.title('Games versus Score')
plt.grid(True)
plt.show()

