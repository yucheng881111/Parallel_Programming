#include <stdio.h>
#include <pthread.h>
#include <immintrin.h>
#include "SIMDxorshift/include/simdxorshift128plus.h"

typedef long long ll;
ll total_num_in_circle = 0;
pthread_mutex_t lock;

void* estimate_pi(void* num){
	// create a new key
	avx_xorshift128plus_key_t mykey;
	avx_xorshift128plus_init(324,4444,&mykey); // values 324, 4444 are arbitrary, must be non-zero
	ll tosses = *(ll*)num;
	ll in_circle = 0;
	// Initializes 256-bit vector with float32 values
	__m256 unit = _mm256_set_ps(INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX);

	for(int i=0;i<tosses;i+=8){
		// generate 32 random bytes, do this as many times as you want
		__m256i randomstuff = avx_xorshift128plus(&mykey);
		// Converts extended packed 32-bit integer values to packed single-precision floating point values
		__m256 x = _mm256_cvtepi32_ps(randomstuff);
		// Divides float32 vectors
		__m256 unit_x = _mm256_div_ps(x, unit);

		// same method to calc y
		randomstuff = avx_xorshift128plus(&mykey);
		__m256 y = _mm256_cvtepi32_ps(randomstuff);
		__m256 unit_y = _mm256_div_ps(y, unit);

		// calc x^2 + y^2
		// Multiplies float32 vectors
		__m256 res = _mm256_add_ps(_mm256_mul_ps(unit_x, unit_x), _mm256_mul_ps(unit_y, unit_y)); // x^2 + y^2

		float tmp[8];
		// Moves packed single-precision floating point values from a float32 vector to an aligned memory location
		_mm256_store_ps(tmp, res);
		for(int j=0;j<8;++j){
	    		if(tmp[j] <= 1.f){
				in_circle++;
			};
		}
	}

	pthread_mutex_lock(&lock);
    	total_num_in_circle += in_circle;
    	pthread_mutex_unlock(&lock);
}


int main(int argc, char** argv){
	int num_of_threads = atoi(argv[1]);
	ll total_num_of_tosses = atoll(argv[2]);
	pthread_t* threads = (pthread_t*) malloc(num_of_threads * sizeof(pthread_t));
    	ll num_tosses = total_num_of_tosses / num_of_threads;
	ll num_tosses_left = total_num_of_tosses % num_of_threads;
    	pthread_mutex_init(&lock, NULL);

	for(int i=0;i<num_of_threads;++i){
		if(i == 0){
			ll t = num_tosses + num_tosses_left;
			pthread_create(&threads[i], NULL, estimate_pi, (void*)&t);
		}else{
			pthread_create(&threads[i], NULL, estimate_pi, (void*)&num_tosses);
		}
	}
	for(int i=0;i<num_of_threads;++i){
		pthread_join(threads[i], NULL);
	}

	free(threads);
    	pthread_mutex_destroy(&lock);
    	float pi = 4 * (float)total_num_in_circle / total_num_of_tosses;
    	printf("%.6f\n", pi);

return 0;
}
