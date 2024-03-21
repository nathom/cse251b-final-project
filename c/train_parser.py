import matplotlib.pyplot as plt

# Define lists to store data
games = []
percent_1024 = []
percent_2048 = []
percent_4096 = []
percent_8192 = []
percent_8192 = []

# Read data from file
with open('train_data.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        # Check if the line contains data regarding games
        if "avg" in line:
            # Extract number of games
            games.append(int(line.split()[0]))
        elif '1024' in line:
            # Extract percentage achieved for 2048 tile
            percent_1024.append(float(line.split()[1].strip('%')))
            while (len(percent_1024) < len(games)):
                percent_1024.append(0)
        elif '2048' in line:
            # Extract percentage achieved for 2048 tile
            percent_2048.append(float(line.split()[1].strip('%')))
            while (len(percent_2048) < len(games)):
                percent_2048.append(0)
        elif '4096' in line:
            # Extract percentage achieved for 4096 tile
            percent_4096.append(float(line.split()[1].strip('%')))
            while (len(percent_4096) < len(games)):
                percent_4096.append(0)
        elif '8192' in line:
            # Extract percentage achieved for 4096 tile
            percent_8192.append(float(line.split()[1].strip('%')))
            while (len(percent_8192) < len(games)):
                percent_8192.append(0)


# Plotting
plt.plot(games, percent_1024, marker='o', label='1024')
plt.plot(games, percent_2048, marker='o', label='2048')
plt.plot(games, percent_4096, marker='o', label='4096')
plt.plot(games, percent_8192, marker='o', label='8192')
# plt.plot(games, percent_16384, marker='o', label='8192')

# Adding labels and title
plt.xlabel('Simulated Games')
plt.ylabel('Tile achieved')
plt.title('Games Simulated vs Maximum Tile Achieved')
plt.legend()

# Displaying the plot
plt.grid(True)
plt.show()

