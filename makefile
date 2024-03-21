fast:
	gcc ./c/2048.c -o ./binary/2048.exe
	./binary/2048.exe mc 50 100 false

plot:
	python ./data/mcts_plots.py