import tkinter as tk

def read_iterations(file_path):
    with open(file_path, 'r') as file:
        iterations = []
        current_iteration = []

        for line in file:
            line = line.strip()
            if line.startswith("Iteration"):
                if current_iteration:
                    iterations.append(current_iteration)
                    current_iteration = []
            elif line:
                current_iteration.append([int(x.strip("[]'))")) for x in line.split(',')])
        
        # Append the last iteration
        if current_iteration:
            iterations.append(current_iteration)
        
        return iterations

def draw_board(canvas, iteration):
    # Colors for tiles
    colors = {
        0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
        16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
        256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"
    }

    total_score = 0

    for row_idx, row in enumerate(iteration):
        for col_idx, value in enumerate(row):
            tile_x0 = col_idx * TILE_SIZE
            tile_y0 = row_idx * TILE_SIZE
            tile_x1 = tile_x0 + TILE_SIZE
            tile_y1 = tile_y0 + TILE_SIZE
            tile_color = colors.get(value, "#ccc0b3")
            canvas.create_rectangle(tile_x0, tile_y0, tile_x1, tile_y1, fill=tile_color, outline="#ccc0b3")
            if value != 0:
                canvas.create_text((tile_x0 + tile_x1) // 2, (tile_y0 + tile_y1) // 2, text=str(value), font=("Arial", 24, "bold"), fill="black")
            total_score += value
    
    return total_score

def main():
    file_path = "iterations.txt"
    iterations = read_iterations(file_path)

    root = tk.Tk()
    root.title("2048")

    # Calculate board size based on the number of rows and columns
    rows = len(iterations[0])
    cols = len(iterations[0][0])
    global TILE_SIZE
    TILE_SIZE = 100
    canvas_width = cols * TILE_SIZE
    canvas_height = rows * TILE_SIZE
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack()

    # Add labels for total score and iteration count
    total_score_label = tk.Label(root, text="Total Score: 0", font=("Arial", 16))
    total_score_label.pack()
    iteration_label = tk.Label(root, text="Iteration: 0", font=("Arial", 16))
    iteration_label.pack()

    total_score = 0
    for i, iteration in enumerate(iterations):
        canvas.delete("all")
        total_score = draw_board(canvas, iteration)
        # canvas.create_text(canvas_width // 2, 20, text=f"Total Score: {total_score}", font=("Arial", 16), fill="black")
        total_score_label.config(text=f"Total Score: {total_score}")
        iteration_label.config(text=f"Iteration: {i + 1}")
        root.update()
        root.update()
        # Adjust the speed of displaying iterations (in milliseconds)
        root.after(100)

    root.mainloop()

if __name__ == "__main__":
    main()
