import matplotlib.pyplot as plt
import sys

# Define lists to store data
games = []
percent_1024 = []
percent_2048 = []
percent_4096 = []
percent_8192 = []
percent_8192 = []
percent_16384 = []

# Read data from file
with open(sys.argv[1], 'r') as file:
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
        elif '16384' in line:
            # Extract percentage achieved for 4096 tile
            percent_16384.append(float(line.split()[1].strip('%')))
            while (len(percent_16384) < len(games)):
                percent_16384.append(0)

max_len = max(len(games), len(percent_1024), len(percent_2048), len(percent_4096), len(percent_8192))
percent_1024 += [percent_1024[-1]] * (max_len - len(percent_1024))
percent_2048 += [percent_2048[-1]] * (max_len - len(percent_2048))
percent_4096 += [percent_4096[-1]] * (max_len - len(percent_4096))
percent_8192 += [percent_8192[-1]] * (max_len - len(percent_8192))
percent_16384 += [percent_16384[-1]] * (max_len - len(percent_16384))

# Plotting
plt.plot(games, percent_1024, label='1024')
plt.plot(games, percent_2048, label='2048')
plt.plot(games, percent_4096, label='4096')
plt.plot(games, percent_8192, label='8192')
plt.plot(games, percent_16384, label='16384')

# Adding labels and title
plt.xlabel('Simulated Games')
plt.ylabel('Percent Games')
plt.title('Games Simulated vs Percent Games Reached Tile')
plt.legend()

# Displaying the plot
plt.grid(True)
plt.show()

