// Copyright 2026 The Stars Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"

	"github.com/pointlander/gradient"
	"github.com/pointlander/stars/kmeans"
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures  []float64
	Embedding []float64
	Label     string
	Cluster   int
	Index     int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

// standardizeRows centers each column and scales it to unit variance.
func standardizeRows(x []float64, n, size int) {
	if n == 0 || size == 0 {
		return
	}
	mean := make([]float64, size)
	for i := 0; i < n; i++ {
		row := x[i*size : (i+1)*size]
		for j, v := range row {
			mean[j] += v
		}
	}
	invN := 1 / float64(n)
	for j := range mean {
		mean[j] *= invN
	}
	std := make([]float64, size)
	for i := 0; i < n; i++ {
		row := x[i*size : (i+1)*size]
		for j, v := range row {
			d := v - mean[j]
			std[j] += d * d
		}
	}
	for j := range std {
		std[j] = math.Sqrt(std[j] * invN)
		if std[j] < 1e-12 {
			std[j] = 1
		}
	}
	for i := 0; i < n; i++ {
		row := x[i*size : (i+1)*size]
		for j := range row {
			row[j] = (row[j] - mean[j]) / std[j]
		}
	}
}

// jacobiEigen returns the eigenvalues and row-major eigenvectors of a
// symmetric n×n matrix. Eigenvector k is column k of vectors.
func jacobiEigen(a []float64, n int) (diag []float64, vectors []float64) {
	values := append([]float64(nil), a...)
	vectors = make([]float64, n*n)
	for i := 0; i < n; i++ {
		vectors[i*n+i] = 1
	}
	for sweep := 0; sweep < 64; sweep++ {
		off := 0.0
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				v := values[p*n+q]
				off += v * v
			}
		}
		if off <= 1e-24 {
			break
		}
		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				apq := values[p*n+q]
				if apq == 0 {
					continue
				}
				app := values[p*n+p]
				aqq := values[q*n+q]
				tau := (aqq - app) / (2 * apq)
				t := math.Copysign(1/(math.Abs(tau)+math.Sqrt(1+tau*tau)), tau)
				c := 1 / math.Sqrt(1+t*t)
				s := t * c
				for k := 0; k < n; k++ {
					if k == p || k == q {
						continue
					}
					aik := c*values[k*n+p] - s*values[k*n+q]
					akq := s*values[k*n+p] + c*values[k*n+q]
					values[k*n+p] = aik
					values[p*n+k] = aik
					values[k*n+q] = akq
					values[q*n+k] = akq
				}
				values[p*n+p] = c*c*app - 2*s*c*apq + s*s*aqq
				values[q*n+q] = s*s*app + 2*s*c*apq + c*c*aqq
				values[p*n+q] = 0
				values[q*n+p] = 0
				for k := 0; k < n; k++ {
					vip := c*vectors[k*n+p] - s*vectors[k*n+q]
					viq := s*vectors[k*n+p] + c*vectors[k*n+q]
					vectors[k*n+p] = vip
					vectors[k*n+q] = viq
				}
			}
		}
	}
	diag = make([]float64, n)
	for i := 0; i < n; i++ {
		diag[i] = values[i*n+i]
	}
	return diag, vectors
}

// leftSingularRows returns the leading left singular vectors of an n×size
// row-major matrix, stored as n rows of length width. Components past the
// numerical rank are zero.
func leftSingularRows(x []float64, n, size, width int) []float64 {
	emb := make([]float64, n*width)
	if n == 0 || size == 0 || width == 0 {
		return emb
	}
	cov := make([]float64, size*size)
	for i := 0; i < n; i++ {
		row := x[i*size : (i+1)*size]
		for a := 0; a < size; a++ {
			for b := 0; b < size; b++ {
				cov[a*size+b] += row[a] * row[b]
			}
		}
	}
	diag, vectors := jacobiEigen(cov, size)
	order := make([]int, size)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		return diag[order[i]] > diag[order[j]]
	})
	maxSigma := 0.0
	if diag[order[0]] > 0 {
		maxSigma = math.Sqrt(diag[order[0]])
	}
	rank := 0
	for _, comp := range order {
		if rank >= width {
			break
		}
		sigma := 0.0
		if diag[comp] > 0 {
			sigma = math.Sqrt(diag[comp])
		}
		if sigma <= maxSigma*1e-10 {
			break
		}
		for i := 0; i < n; i++ {
			dot := 0.0
			row := x[i*size : (i+1)*size]
			for a := 0; a < size; a++ {
				dot += row[a] * vectors[a*size+comp]
			}
			emb[i*width+rank] = dot / sigma
		}
		rank++
	}
	return emb
}

// lossValue evaluates a scalar loss without backpropagating.
func lossValue(loss gradient.Meta[float64]) float64 {
	var value float64
	loss(func(v *gradient.V[float64]) bool {
		if len(v.X) > 0 {
			value = v.X[0]
		}
		return true
	})
	return value
}

// samePartition reports whether two labelings are the same up to label ids.
func samePartition(a, b []int, k int) bool {
	if len(a) != len(b) {
		return false
	}
	forward := make([]int, k)
	reverse := make([]int, k)
	for i := range forward {
		forward[i] = -1
		reverse[i] = -1
	}
	for i := range a {
		ai, bi := a[i], b[i]
		if ai < 0 || ai >= k || bi < 0 || bi >= k {
			return false
		}
		if forward[ai] == -1 {
			forward[ai] = bi
		} else if forward[ai] != bi {
			return false
		}
		if reverse[bi] == -1 {
			reverse[bi] = ai
		} else if reverse[bi] != ai {
			return false
		}
	}
	return true
}

// LearnEmbeddingIris learns the iris embedding.
// Measurements are centered and scaled. The embedding starts at the leading
// left singular vectors of that matrix, then training fits a column-dropped
// Gram matrix with a zero diagonal so each flower is rebuilt from the others.
func LearnEmbeddingIris(iris []Fisher, size, width, iterations int) []Fisher {
	const (
		Eta    = 1e-3
		adamB1 = 0.9
		adamB2 = 0.999
	)
	rng := rand.New(rand.NewSource(1))
	context := gradient.Context[float64]{}
	length := len(iris)
	cp := make([]Fisher, length)
	copy(cp, iris)
	set := context.NewSet()
	set.Add("i", width, length)
	set.AddData("x", size, length)
	set.AddData("m", length, length)
	set.InitAdam(rng)
	x, index := set.ByName["x"], 0
	for _, row := range iris {
		for _, value := range row.Measures {
			x.X[index] = value
			index++
		}
	}
	standardizeRows(x.X, length, size)
	mask := set.ByName["m"]
	for r := 0; r < length; r++ {
		row := mask.X[r*length : (r+1)*length]
		for c := range row {
			if r != c {
				row[c] = 1
			}
		}
	}
	embedding := set.ByName["i"]
	copy(embedding.X, leftSingularRows(x.X, length, size, width))

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	T := context.U(context.T)
	Mul := context.B(context.Mul)
	Hadamard := context.B(context.Hadamard)
	Dropout := context.U(context.Dropout)
	Square := context.U(context.Square)
	Avg := context.U(context.Avg)
	Quadratic := context.B(context.Quadratic)

	gram := Hadamard(Dropout(Square(set.Get("i")), dropout), set.Get("m"))
	sa := T(Mul(gram, T(set.Get("x"))))
	loss := Avg(Quadratic(set.Get("x"), sa))
	gramEval := Hadamard(Square(set.Get("i")), set.Get("m"))
	saEval := T(Mul(gramEval, T(set.Get("x"))))
	lossEval := Avg(Quadratic(set.Get("x"), saEval))

	finite := func(v float64) bool {
		return !math.IsNaN(v) && !math.IsInf(v, 0)
	}
	best := append([]float64(nil), embedding.X...)
	bestEval := lossValue(lossEval)
	fmt.Println("init", bestEval)
	for step := 0; step < iterations; step++ {
		set.Zero()
		train := gradient.Gradient(loss).X[0]
		if !finite(train) {
			fmt.Println("non-finite loss", step, train)
			break
		}
		set.Adam(adamB1, adamB2, Eta)
		eval := lossValue(lossEval)
		if !finite(eval) {
			fmt.Println("non-finite eval", step, eval)
			break
		}
		if eval < bestEval {
			bestEval = eval
			copy(best, embedding.X)
		}
		if step%128 == 0 || step == iterations-1 {
			fmt.Println("step", step, "train", train, "eval", eval)
		}
	}
	copy(embedding.X, best)
	fmt.Println("best", bestEval)

	vectors := make([][]float64, length)
	for i := range vectors {
		vectors[i] = append([]float64(nil), embedding.X[i*width:(i+1)*width]...)
		cp[i].Embedding = vectors[i]
	}
	const (
		k             = 3
		consensusRuns = 33
	)
	meta := make([][]float64, length)
	for i := range meta {
		meta[i] = make([]float64, length)
	}
	labelings := make([][]int, 0, consensusRuns)
	for run := 0; run < consensusRuns; run++ {
		clusters, _, err := kmeans.Kmeans(int64(run+1), vectors, k, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		labelings = append(labelings, clusters)
		for i := 0; i < length; i++ {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					meta[i][j]++
				}
			}
		}
	}
	stable := true
	for run := 1; run < len(labelings); run++ {
		if !samePartition(labelings[0], labelings[run], k) {
			stable = false
			break
		}
	}
	clusters := labelings[0]
	if stable {
		fmt.Println("clusters", "stable")
	} else {
		fmt.Println("clusters", "consensus")
		scale := 1 / float64(consensusRuns)
		for i := range meta {
			for j := range meta[i] {
				meta[i][j] *= scale
			}
		}
		var err error
		clusters, _, err = kmeans.Kmeans(1, meta, k, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
	}
	for i := range clusters {
		cp[i].Cluster = clusters[i]
	}
	sort.Slice(cp, func(i, j int) bool {
		return cp[i].Cluster < cp[j].Cluster
	})
	return cp
}

// ClusterMode is the iris clustering mode
func ClusterMode() {
	rng := rand.New(rand.NewSource(1))
	iris := Load()
	rng.Shuffle(len(iris), func(i, j int) {
		iris[i], iris[j] = iris[j], iris[i]
	})
	embedding := LearnEmbeddingIris(iris, 4, 5, 2*1024)
	accuracy := make(map[string][3]int)
	for i := range embedding {
		fmt.Println(embedding[i].Cluster, embedding[i].Label)
		counts := accuracy[embedding[i].Label]
		counts[embedding[i].Cluster]++
		accuracy[embedding[i].Label] = counts
	}
	for i, v := range accuracy {
		fmt.Println(i, v)
	}
}

func main() {
	ClusterMode()
}
