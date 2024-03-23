fast:
	gcc ./c/2048.c -o ./binary/2048.exe
	./binary/2048.exe random 0 10 false

tuple:
	g++ -o ./binary/tuple.exe ./c/2048_tuple_network.cpp -std=c++20
	./binary/tuple.exe random false 1000 100

plot:
	python ./data/mcts_plots.py