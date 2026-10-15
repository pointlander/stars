package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestLoadBcwd(t *testing.T) {
	records := LoadBcwd()
	if len(records) != 569 {
		t.Fatalf("rows %d", len(records))
	}
	counts := map[string]int{}
	for i, record := range records {
		if len(record.Measures) != 30 {
			t.Fatalf("row %d measures %d", i, len(record.Measures))
		}
		if record.Label != "M" && record.Label != "B" {
			t.Fatalf("row %d label %q", i, record.Label)
		}
		counts[record.Label]++
	}
	if counts["M"] != 212 || counts["B"] != 357 {
		t.Fatalf("labels %v", counts)
	}
}

func TestBcwdClusterAccuracy(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	records := LoadBcwd()
	rng.Shuffle(len(records), func(i, j int) {
		records[i], records[j] = records[j], records[i]
	})
	got := clusterStandardized(records, 2, 33)
	if len(got) != len(records) {
		t.Fatalf("len %d", len(got))
	}
	var agree0, agree1 int
	for _, record := range got {
		if record.Cluster < 0 || record.Cluster > 1 {
			t.Fatalf("cluster %d", record.Cluster)
		}
		malignant := record.Label == "M"
		if malignant == (record.Cluster == 0) {
			agree0++
		}
		if malignant == (record.Cluster == 1) {
			agree1++
		}
	}
	agree := agree0
	if agree1 > agree {
		agree = agree1
	}
	accuracy := float64(agree) / float64(len(got))
	if accuracy < 0.90 {
		t.Fatalf("accuracy %g", accuracy)
	}
}

func TestBcwdOneStep(t *testing.T) {
	records := LoadBcwd()
	got := LearnEmbeddingIris(records, 30, 5, 2, 1)
	if len(got) != len(records) {
		t.Fatalf("len %d", len(got))
	}
	for i, record := range got {
		if len(record.Embedding) != 5 {
			t.Fatalf("row %d embedding %d", i, len(record.Embedding))
		}
		if record.Cluster < 0 || record.Cluster > 1 {
			t.Fatalf("row %d cluster %d", i, record.Cluster)
		}
		if record.Label != "M" && record.Label != "B" {
			t.Fatalf("row %d label %q", i, record.Label)
		}
	}
}

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
