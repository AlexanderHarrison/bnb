HIGHSLIBLOC = HiGHS/build/lib
HIGHSLIB = $(HIGHSLIBLOC)/libhighs.so.1.5.3
HIGHSFLAGS = -IHiGHS/src/ -IHiGHS/build -L$(HIGHSLIBLOC) -lhighs
SRC = c/bnb.cpp c/heap.h c/node.h c/solve.h c/sparse_mat.h c/data.h

bnb.so: $(SRC) $(HIGHSLIB)
	g++ -Wall -Wextra -O3 -lm c/bnb.cpp -shared $(HIGHSFLAGS) -o bnb.so

bnb: $(SRC) $(HIGHSLIB)
	g++ -Wall -Wextra -O3 -g -lm $(HIGHSFLAGS) c/bnb.cpp -o bnb

$(HIGHSLIB): HiGHS/src/Highs.h
	cmake --build HiGHS/build

HiGHS/src/Highs.h:
	git clone --depth 1 --branch v1.5.3 https://github.com/ERGO-Code/HiGHS.git
	cmake -DBUILD_SHARED_LIBS=OFF -S HiGHS -B HiGHS/build
