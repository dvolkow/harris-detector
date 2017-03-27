all: harris_algorithm

EasyBMP.o: \
	../EasyBMP/EasyBMP.cpp \
	../EasyBMP/EasyBMP.h \
	../EasyBMP/EasyBMP_BMP.h \
	../EasyBMP/EasyBMP_DataStructures.h \
	../EasyBMP/EasyBMP_VariousBMPutilities.h
	$(CXX) -g -c $< -o $@ -O3
libeasybmp.a: EasyBMP.o
	ar -rcs $@ $<
harris_algorithm.o: harris_algorithm.cu
	nvcc -std=c++11 -c -IEasyBMP $< -o $@
harris_algorithm: harris_algorithm.o libeasybmp.a
	nvcc -O3 -std=c++11 $< -o $@ -L. -leasybmp
clean:
	rm -rf *.o libeasybmp.a harris_algorithm
