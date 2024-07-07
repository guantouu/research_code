.SECONDEXPANSION:

MAINS := main.cpp
ALLSRC := $(wildcard *.cpp)
SRC := $(filter-out $(MAINS),$(ALLSRC))
ALLOBJ := $(ALLSRC:.cpp=.o)
OBJ := $(SRC:.cpp=.o)

CXX := g++
CXXFLAGS := -fopenmp -O3 -std=c++0x -w -g  # 添加-g标志
LDFLAGS := -fopenmp -O3 -std=c++0x -w -g   # 链接时也添加-g标志

.PHONY: all clean debug
all: $(MAINS:.cpp=)

$(MAINS:.cpp=): $(OBJ) $$@.o
	$(CXX) $(LDFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

depend: .depend
.depend: $(ALLSRC)
	@$(RM) .depend
	@$(CXX) -MM $(CXXFLAGS) $^ > .depend;
include .depend

clean:
	$(RM) $(MAINS:.cpp=)
	$(RM) $(ALLOBJ)


