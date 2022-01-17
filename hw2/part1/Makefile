pi.out : pi.c SIMDxorshift/src/simdxorshift128plus.c SIMDxorshift/src/xorshift128plus.c
	gcc -fPIC -std=c99 -O3 -I. -I./SIMDxorshift/include -mavx2 -march=native -Wextra -pedantic -Wshadow $^ -o $@ -lpthread
