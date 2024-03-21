fast:
	gcc ./c/2048.c -o ./binary/2048.exe
	./binary/2048.exe mc 50 100 false

tuple:
	g++ -o ./binary/tuple.exe ./c/2048_tuple_network.cpp
	./binary/tuple.exe tuple false 1000 5

plot:
	python ./data/mcts_plots.py