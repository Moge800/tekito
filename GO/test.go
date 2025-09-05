package main

import (
	"fmt"
	"math"
)

func isPrime(n int) bool {
	if n < 2 {
		return false
	}
	sqrtN := int(math.Sqrt(float64(n)))
	for i := 2; i <= sqrtN; i++ {
		if n%i == 0 {
			return false
		}
	}
	return true
}

func calc() {
	max := 1000000 // ここを増やすとさらに重くなります
	var lastPrime int
	for i := 2; i <= max; i++ {
		if isPrime(i) {
			lastPrime = i
		}
	}
	fmt.Printf("最後に見つかった素数: %d\n", lastPrime)
}
