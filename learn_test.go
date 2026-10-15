package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestAutoencoderEmbedding(t *testing.T) {
	iris := Load()
	const size, width, iterations = 4, 5, 256
	n := len(iris)
	if n == 0 {
		t.Fatal("empty iris")
	}
	x := make([]float64, 0, n*size)
	for _, row := range iris {
		x = append(x, row.Measures...)
	}
	standardized := append([]float64(nil), x...)
	standardizeColumns(standardized, n, size)

	rng := rand.New(rand.NewSource(1))
	emb, first, last := autoencoderEmbedding(x, n, size, width, iterations, rng)
	if len(emb) != n*width {
		t.Fatalf("embedding length %d", len(emb))
	}
	for k := 0; k < width; k++ {
		mean := 0.0
		for i := 0; i < n; i++ {
			v := emb[i*width+k]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("component %d sample %d is %g", k, i, v)
			}
			mean += v
		}
		mean /= float64(n)
		variance := 0.0
		for i := 0; i < n; i++ {
			d := emb[i*width+k] - mean
			variance += d * d
		}
		variance /= float64(n)
		if variance < 1e-4 {
			t.Fatalf("component %d variance %g", k, variance)
		}
	}
	if !(last < first) {
		t.Fatalf("loss did not fall: %g -> %g", first, last)
	}
	meanLoss := 0.0
	for i := 0; i < n; i++ {
		sum := 0.0
		for f := 0; f < size; f++ {
			v := standardized[i*size+f]
			sum += v * v
		}
		meanLoss += 0.5 * sum
	}
	meanLoss /= float64(n)
	if !(last < meanLoss) {
		t.Fatalf("final loss %g is not below the mean predictor %g", last, meanLoss)
	}
}
