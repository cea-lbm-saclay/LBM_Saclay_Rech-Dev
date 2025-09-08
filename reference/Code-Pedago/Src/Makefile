EXEC    = ../Bin/LBM_Bin
DIR_OBJ = ./obj90
DIR_SRC = .

COMPIL  = gfortran

MAIN    = LBM_Main

OBJETS  = LBM_Lib_D2Q5 LBM_Input Equation_Transport LBM_Lib_Util

OBJ_ALL = $(OBJETS:%=$(DIR_OBJ)/%.o)

OPTIONS = -Wall -Wtabs -g -O3

$(EXEC) : $(MAIN).o $(OBJ_ALL) 
	$(COMPIL) $(OPTIONS) -o $(EXEC) $(DIR_OBJ)/$(MAIN).o \
				  $(OBJ_ALL)

$(MAIN).o : $(MAIN).f
	$(COMPIL) $(OPTIONS) -c -o  $(DIR_OBJ)/$(MAIN).o $(MAIN).f
$(DIR_OBJ)/%.o :$(DIR_SRC)/%.f
	$(COMPIL) $(OPTIONS) -c -o $@ $<


clean:
	rm -f *.f~ makefile~ *.scr~
	rm -f obj90/*.o

