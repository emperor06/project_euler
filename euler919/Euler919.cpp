/**
 * Project Euler #919 - Fortunate Triangles
 * 
 * ABC, a triangle of sides (a, b, c). H is the orthocenter, r is the circumradius.
 * We want (a, b, c) integers with a<=b<=c, and either 2AH = r, or 2BH = r, or 2CH = r.
 * 
 * AH = 2r.|cos(A)| = 2r. |b² + c² - a²| / (2bc)
 * 2AH = r <=> 2.|b² + c² - a²| = bc (same for the other angles)
 * If that's the case, then |cos(A)| = |b² + c² - a²| / (2bc) = (bc/2) / (2bc) = 1/4
 * <=> A = |arccos(1/4)| <=> A is either 72.52° or 104.48°
 * 
 * It's then easy to prove the following assertions. A fortunate triangle:
 * - has integer lengths with an angle of either 72.52° or 104.48°
 * - cannot be a right triangle
 * - cannot be isoceles unless it is congruent with (1, 2, 2)
 * - a < b
 * - A, being the angle opposite the smallest side, cannot be 72° or 104°
 * 
 * This algorithm chooses a and b, then computes c so that the angle is 72 or 104,
 * then checks if c is an integer. If that's the case, (a, b, c) is fortunate.
 * 
 * by Draxar (2024)
 */

#include <stdio.h>
#include <bit>
#include <cstdint>
#include <cmath>
#include <thread>
#include <atomic>
#include <vector>

constexpr auto BATCH_SIZE = 128;

// Forward declarations
int32_t isSquare(uint64_t);
void work(std::atomic<int64_t>&, std::atomic<int32_t>&, const int32_t, const int64_t);

int main(int argc, char** argv) {
	if (argc <= 1) {
		printf("Usage: %s <maxperimeter> [nbthreads]\n", argv[0]);
		return 0;
	}
	const int64_t  P         = atoi(argv[1]);
	const uint32_t maxcores  = std::thread::hardware_concurrency(); // this can return 0 …
	const uint32_t NT        = argc > 2 ? atoi(argv[2]) : maxcores;
	const int32_t  totalJobs = (P/3 - 1) / BATCH_SIZE + 1;

	std::atomic<int64_t> result{ 0 };
	std::atomic<int32_t> job   { 0 };
	std::vector<std::thread> v;

	// Start threads
	for (uint32_t i = 0; i < NT; i++)
		v.push_back(std::thread{ work, std::ref(result), std::ref(job), totalJobs, P });

	// Track progress if the computation takes (very roughly) more than 2s
	if (P > 100000) {
		while (1) {
			int32_t j = job.load();
			if (j >= totalJobs) break;
			printf("\rProgress: %.2f %%", j * 100. / totalJobs);
			fflush(stdout);
			std::this_thread::sleep_for(std::chrono::seconds(2));
		}
		printf("\rProgress: 100%%   \n");
	}

	// Ensure all threads are done and kill them
	for (uint32_t i = 0; i < NT; i++)
		v[i].join();

	// Harvest and profit
	int64_t r = result.load();
	printf("[Main] result = %lld\n", r);
	return 0;
}

void work(std::atomic<int64_t>& res, std::atomic<int32_t>& job, const int32_t totalJobs, const int64_t P) {
	int64_t sum = 0;

	while (1) {
		const int32_t myjob = job.fetch_add(1, std::memory_order_relaxed);
		if (myjob >= totalJobs) break; // we're done

		const int64_t istart = myjob * BATCH_SIZE + 1;
		const int64_t iend   = (istart + BATCH_SIZE - 1) > P/3 ? P/3 : istart + BATCH_SIZE - 1;

		for (int64_t i = istart; i <= iend; i++) {

			int64_t minj = i + 1;
			int64_t pmi  = P - i;
			int64_t maxj = P - 2*i - 1; // careful, j can play k
			int64_t inc  = i & 1 ? 2 : 1;
			int64_t i2   = i * i;

			for (int64_t j = minj; j <= maxj; j += inc) {

				int64_t  j2    = j*j;
				int64_t  ij    = i*j;
				int64_t  tmp   = pmi - j;
				uint64_t maxk2 = tmp * tmp;
				uint64_t x     = i2 + j2 + ij / 2;

				if (x <= maxk2 && isSquare(x)) {
					int64_t k = std::sqrt(x);
					sum += i + j + k;
				}
				else {
					uint64_t y = i2 + j2 - ij / 2;
					if (y <= maxk2 && isSquare(y)) {
						int64_t k = (int64_t)std::sqrt(y);
						sum += i + j + k;
					}
				}
			}
		}
	}

	res += sum;
}

// isSquare() by Maaartinus (https://stackoverflow.com/a/18686659/3357680)
int32_t isSquare(uint64_t x) {
	static int64_t goodMask = 0xC840C04048404040;
	if (goodMask << x >= 0) return 0;
	const int32_t tz = std::countr_zero(x);
	if ((tz & 1) != 0) return 0;
	x >>= tz;
	if ((x & 7) != 1 || x <= 0) return x == 0;
	// fallback if all fast tests failed
	const int64_t tst = (int64_t)std::sqrt(x);
	return tst * tst == x;
}
