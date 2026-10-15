package kmeans

import (
	"math"
	"testing"
)

func TestMahalanobisIdentityMatchesEuclidean(t *testing.T) {
	first := []float64{1, -2, 0.5}
	second := []float64{-1, 2, 1.5}
	inv := [][]float64{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}
	got, err := MahalanobisDistance(first, second, inv)
	if err != nil {
		t.Fatal(err)
	}
	want, err := EuclideanDistance(first, second)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("got %g want %g", got, want)
	}
	squared, err := SquaredMahalanobisDistance(inv)(first, second)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(squared-got*got) > 1e-12 {
		t.Fatalf("squared %g", squared)
	}
}

func TestMahalanobisScalesAxes(t *testing.T) {
	origin := []float64{0, 0}
	alongX := []float64{3, 0}
	alongY := []float64{0, 2}
	inv := [][]float64{{1.0 / 25, 0}, {0, 1}}
	dx, err := MahalanobisDistance(origin, alongX, inv)
	if err != nil {
		t.Fatal(err)
	}
	dy, err := MahalanobisDistance(origin, alongY, inv)
	if err != nil {
		t.Fatal(err)
	}
	if !(dx < dy) {
		t.Fatalf("scaled x %g is not nearer than y %g", dx, dy)
	}
	ex, _ := EuclideanDistance(origin, alongX)
	ey, _ := EuclideanDistance(origin, alongY)
	if !(ey < ex) {
		t.Fatalf("euclidean setup y %g x %g", ey, ex)
	}
}

func TestCovarianceAndInverse(t *testing.T) {
	data := [][]float64{{0, 0}, {2, 0}, {0, 2}, {2, 2}}
	cov, err := Covariance(data)
	if err != nil {
		t.Fatal(err)
	}
	want := 4.0 / 3
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected := 0.0
			if i == j {
				expected = want
			}
			if math.Abs(cov[i][j]-expected) > 1e-12 {
				t.Fatalf("cov[%d][%d] %g", i, j, cov[i][j])
			}
		}
	}
	inv, err := InvertSymmetric([][]float64{{4, 2}, {2, 3}})
	if err != nil {
		t.Fatal(err)
	}
	wantInv := [][]float64{{0.375, -0.25}, {-0.25, 0.5}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(inv[i][j]-wantInv[i][j]) > 1e-12 {
				t.Fatalf("inv[%d][%d] %g", i, j, inv[i][j])
			}
		}
	}
	if _, err := InvertSymmetric([][]float64{{1, 1}, {1, 1}}); err == nil {
		t.Fatal("singular matrix inverted")
	}
}

func TestKmeansMahalanobisMatchesEuclideanOnIdentity(t *testing.T) {
	data := [][]float64{
		{0, 0}, {0.1, 0}, {0, 0.1},
		{5, 5}, {5.1, 5}, {5, 5.1},
	}
	euclidean, _, err := Kmeans(1, data, 2, SquaredEuclideanDistance, 100)
	if err != nil {
		t.Fatal(err)
	}
	inv := [][]float64{{1, 0}, {0, 1}}
	mahalanobis, _, err := Kmeans(1, data, 2, SquaredMahalanobisDistance(inv), 100)
	if err != nil {
		t.Fatal(err)
	}
	for i := range euclidean {
		if euclidean[i] != mahalanobis[i] {
			t.Fatalf("label %d euclidean %d mahalanobis %d", i, euclidean[i], mahalanobis[i])
		}
	}
	parallel := [][]float64{
		{0, 0}, {1, 1}, {2, 2},
		{0, 2}, {1, 3}, {2, 4},
	}
	distance, err := MahalanobisFromData(parallel)
	if err != nil {
		t.Fatal(err)
	}
	cov, err := Covariance(parallel)
	if err != nil {
		t.Fatal(err)
	}
	inv, err = InvertSymmetric(cov)
	if err != nil {
		t.Fatal(err)
	}
	got, err := distance(parallel[0], parallel[3])
	if err != nil {
		t.Fatal(err)
	}
	want, err := mahalanobisSquared(parallel[0], parallel[3], inv)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("from data %g want %g", got, want)
	}
	labels, _, err := Kmeans(1, parallel, 2, distance, 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(labels) != len(parallel) {
		t.Fatalf("labels %d", len(labels))
	}
	for i, label := range labels {
		if label != 0 && label != 1 {
			t.Fatalf("label %d is %d", i, label)
		}
	}
}
