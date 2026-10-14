package main

import (
	"math"
	"testing"
)

func TestJacobiEigen(t *testing.T) {
	// [[2, 1], [1, 2]] has eigenvalues 3 and 1.
	diag, vectors := jacobiEigen([]float64{2, 1, 1, 2}, 2)
	got := []float64{diag[0], diag[1]}
	if got[0] < got[1] {
		got[0], got[1] = got[1], got[0]
	}
	want := []float64{3, 1}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-8 {
			t.Fatalf("eigenvalues %v", diag)
		}
	}
	// Columns are unit length.
	for col := 0; col < 2; col++ {
		norm := math.Hypot(vectors[col], vectors[2+col])
		if math.Abs(norm-1) > 1e-8 {
			t.Fatalf("column %d norm %g", col, norm)
		}
	}
}

func TestLeftSingularRowsProjector(t *testing.T) {
	x := []float64{
		1, 0,
		0, 2,
		3, 0,
		0, 4,
	}
	const n, size, width = 4, 2, 2
	emb := leftSingularRows(x, n, size, width)
	out := make([]float64, len(x))
	for i := 0; i < n; i++ {
		ei := emb[i*width : (i+1)*width]
		for j := 0; j < n; j++ {
			ej := emb[j*width : (j+1)*width]
			dot := 0.0
			for k := 0; k < width; k++ {
				dot += ei[k] * ej[k]
			}
			for f := 0; f < size; f++ {
				out[i*size+f] += dot * x[j*size+f]
			}
		}
	}
	for i, v := range x {
		if math.Abs(out[i]-v) > 1e-6 {
			t.Fatalf("reconstruction %v want %v", out, x)
		}
	}
}

func TestStandardizeRows(t *testing.T) {
	x := []float64{
		1, 3,
		3, 3,
		5, 3,
	}
	standardizeRows(x, 3, 2)
	// Column 0 has mean 3 and population std sqrt(8/3). Column 1 is constant.
	if math.Abs(x[1]) > 1e-12 || math.Abs(x[3]) > 1e-12 || math.Abs(x[5]) > 1e-12 {
		t.Fatalf("constant column changed: %v", x)
	}
	mean, variance := 0.0, 0.0
	for i := 0; i < 3; i++ {
		mean += x[i*2]
		variance += x[i*2] * x[i*2]
	}
	mean /= 3
	variance /= 3
	if math.Abs(mean) > 1e-12 || math.Abs(variance-1) > 1e-12 {
		t.Fatalf("mean %g variance %g values %v", mean, variance, x)
	}
}
