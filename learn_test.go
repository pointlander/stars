package main

import (
	"math"
	"testing"
)

func TestPCARowsIris(t *testing.T) {
	iris := Load()
	const size, width = 4, 5
	n := len(iris)
	if n == 0 {
		t.Fatal("empty iris")
	}
	x := make([]float64, n*size)
	index := 0
	for _, row := range iris {
		copy(x[index:], row.Measures)
		index += size
	}
	emb := pcaRows(x, n, size, width)

	for k := 0; k < width; k++ {
		norm := 0.0
		for i := 0; i < n; i++ {
			v := emb[i*width+k]
			norm += v * v
		}
		if math.Abs(norm-1) > 1e-6 {
			t.Fatalf("component %d norm %g", k, norm)
		}
	}
	constant := 1 / math.Sqrt(float64(n))
	for i := 0; i < n; i++ {
		if math.Abs(emb[i*width+size]-constant) > 1e-6 {
			t.Fatalf("constant component %g", emb[i*width+size])
		}
	}
	for a := 0; a < width; a++ {
		for b := a + 1; b < width; b++ {
			dot := 0.0
			for i := 0; i < n; i++ {
				dot += emb[i*width+a] * emb[i*width+b]
			}
			if math.Abs(dot) > 1e-6 {
				t.Fatalf("components %d and %d dot %g", a, b, dot)
			}
		}
	}

	mean := make([]float64, size)
	for i := 0; i < n; i++ {
		for j := 0; j < size; j++ {
			mean[j] += x[i*size+j]
		}
	}
	for j := range mean {
		mean[j] /= float64(n)
	}
	for i := 0; i < n; i++ {
		for f := 0; f < size; f++ {
			got := 0.0
			for j := 0; j < n; j++ {
				dot := 0.0
				for k := 0; k < size; k++ {
					dot += emb[i*width+k] * emb[j*width+k]
				}
				got += dot * (x[j*size+f] - mean[f])
			}
			want := x[i*size+f] - mean[f]
			if math.Abs(got-want) > 1e-5 {
				t.Fatalf("centered sample %d feature %d got %g want %g", i, f, got, want)
			}
		}
	}
	for i := 0; i < n; i++ {
		for f := 0; f < size; f++ {
			got := 0.0
			for j := 0; j < n; j++ {
				dot := 0.0
				for k := 0; k < width; k++ {
					dot += emb[i*width+k] * emb[j*width+k]
				}
				got += dot * x[j*size+f]
			}
			if math.Abs(got-x[i*size+f]) > 1e-5 {
				t.Fatalf("raw sample %d feature %d got %g want %g", i, f, got, x[i*size+f])
			}
		}
	}
}
