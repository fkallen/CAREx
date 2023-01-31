all: makepairregionbed counteditsinpseudoreads convertgapfilleroutput konnectoraddmateinfo reorderReadsByFai countgapeditsfromsam

makepairregionbed: makepairregionbed.cpp
	g++ -std=c++17 -O3 -Wall makepairregionbed.cpp -o makepairregionbed

counteditsinpseudoreads: counteditsinpseudoreads.cpp
	g++ -std=c++17 -O3 -g -Wall counteditsinpseudoreads.cpp -lz -lpthread -fopenmp -o counteditsinpseudoreads

convertgapfilleroutput: convertgapfilleroutput.cpp
	g++ -std=c++17 -O3 -g -Wall convertgapfilleroutput.cpp -o convertgapfilleroutput

konnectoraddmateinfo: konnectoraddmateinfo.cpp
	g++ -std=c++17 -O3 -g -Wall konnectoraddmateinfo.cpp -lz -lpthread -o konnectoraddmateinfo

reorderReadsByFai: reorderReadsByFai.cpp
	g++ -std=c++17 -O3 -g -Wall reorderReadsByFai.cpp -lz -lpthread -o reorderReadsByFai

countgapeditsfromsam: countgapeditsfromsam.cpp
	g++ -std=c++17 -O3 -g -Wall countgapeditsfromsam.cpp -lz -lpthread -o countgapeditsfromsam


clean:
	rm -f makepairregionbed counteditsinpseudoreads convertgapfilleroutput \
	konnectoraddmateinfo reorderReadsByFai countgapeditsfromsam
