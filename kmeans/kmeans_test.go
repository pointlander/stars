package kmeans

import "testing"

func TestKmeansConverges(t *testing.T) {
	data := [][]float64{
		{0, 0}, {0.1, -0.1}, {-0.2, 0.1},
		{5, 5}, {5.2, 4.8}, {4.7, 5.1},
	}
	labels, _, err := Kmeans(1, data, 2, SquaredEuclideanDistance, -1)
	if err != nil {
		t.Fatal(err)
	}
	if len(labels) != len(data) {
		t.Fatalf("labels %v", labels)
	}
	if labels[0] != labels[1] || labels[0] != labels[2] {
		t.Fatalf("first group split: %v", labels)
	}
	if labels[3] != labels[4] || labels[3] != labels[5] {
		t.Fatalf("second group split: %v", labels)
	}
	if labels[0] == labels[3] {
		t.Fatalf("groups merged: %v", labels)
	}
}

func TestKmeansEmptyCluster(t *testing.T) {
	data := [][]float64{{0, 0}, {0, 0}, {10, 10}, {10, 10}}
	labels, _, err := Kmeans(1, data, 3, SquaredEuclideanDistance, -1)
	if err != nil {
		t.Fatal(err)
	}
	if len(labels) != len(data) {
		t.Fatalf("labels %v", labels)
	}
	for _, label := range labels {
		if label < 0 || label >= 3 {
			t.Fatalf("label %d in %v", label, labels)
		}
	}
}
