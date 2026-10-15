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

// standardizeColumns centers each column and scales it to unit variance.
func standardizeColumns(x []float64, n, size int) {
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

// autoencoderEmbedding trains a feed-forward autoencoder and returns the
// bottleneck activation of each sample. Measurements are centered and scaled.
// The network is input → hidden → width → hidden → input, with tanh on every
// layer except the linear reconstruction.
func autoencoderEmbedding(x []float64, n, size, width, iterations int, rng *rand.Rand) (emb []float64, first, last float64) {
	const (
		hidden = 8
		Eta    = 1e-3
	)
	values := append([]float64(nil), x...)
	standardizeColumns(values, n, size)

	context := gradient.Context[float64]{}
	set := context.NewSet()
	set.Add("enc0", size, hidden)
	set.AddBias("benc0", hidden)
	set.Add("enc1", hidden, width)
	set.AddBias("benc1", width)
	set.Add("dec0", width, hidden)
	set.AddBias("bdec0", hidden)
	set.Add("dec1", hidden, size)
	set.AddBias("bdec1", size)
	set.AddData("x", size, n)
	set.InitAdam(rng)
	copy(set.ByName["x"].X, values)

	Add := context.B(context.Add)
	Mul := context.B(context.Mul)
	TanH := context.U(context.TanH)
	Avg := context.U(context.Avg)
	Quadratic := context.B(context.Quadratic)

	hidden0 := TanH(Add(Mul(set.Get("enc0"), set.Get("x")), set.Get("benc0")))
	encoded := TanH(Add(Mul(set.Get("enc1"), hidden0), set.Get("benc1")))
	hidden1 := TanH(Add(Mul(set.Get("dec0"), encoded), set.Get("bdec0")))
	decoded := Add(Mul(set.Get("dec1"), hidden1), set.Get("bdec1"))
	loss := Avg(Quadratic(set.Get("x"), decoded))

	for step := 0; step < iterations; step++ {
		set.Zero()
		l := gradient.Gradient(loss)
		if step == 0 {
			first = l.X[0]
		}
		last = l.X[0]
		set.Adam(gradient.B1, gradient.B2, Eta)
		fmt.Println(last)
	}

	encoded(func(v *gradient.V[float64]) bool {
		emb = append([]float64(nil), v.X...)
		return true
	})
	return emb, first, last
}

// LearnEmbeddingIris learns the iris embedding
func LearnEmbeddingIris(iris []Fisher, size, width, iterations int) []Fisher {
	const Eta = 1e-3
	rng := rand.New(rand.NewSource(1))
	context := gradient.Context[float64]{}
	length := len(iris)
	cp := make([]Fisher, length)
	copy(cp, iris)
	set := context.NewSet()
	set.Add("i", width, len(cp))
	set.AddData("x", size, len(cp))
	set.InitAdam(rng)
	x, index := set.ByName["x"], 0
	for _, row := range iris {
		for _, value := range row.Measures {
			x.X[index] = value
			index++
		}
	}
	i := set.ByName["i"]
	emb, _, _ := autoencoderEmbedding(x.X, length, size, width, iterations, rng)
	copy(i.X, emb)
	for ii := range i.X {
		i.X[ii] *= .01
	}

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	T := context.U(context.T)
	Mul := context.B(context.Mul)
	Dropout := context.U(context.Dropout)
	Square := context.U(context.Square)
	Avg := context.U(context.Avg)
	Quadratic := context.B(context.Quadratic)

	sa := T(Mul(Dropout(Square(set.Get("i")), dropout), T(set.Get("x"))))
	loss := Avg(Quadratic(set.Get("x"), sa))

	for range iterations {
		set.Zero()
		l := gradient.Gradient(loss)
		set.Adam(gradient.B1, gradient.B2, Eta)
		fmt.Println(l.X[0])
	}

	meta := make([][]float64, len(cp))
	for i := range meta {
		meta[i] = make([]float64, len(cp))
	}
	const k = 3

	{
		y := set.ByName["i"]
		vectors := make([][]float64, len(cp))
		for i := range vectors {
			row := make([]float64, width)
			for ii := range row {
				row[ii] = y.X[i*width+ii]
			}
			vectors[i] = row
		}
		for i := 0; i < 33; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), vectors, k, kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := 0; i < len(meta); i++ {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta[i][j]++
					}
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range clusters {
		cp[i].Cluster = clusters[i]
	}
	for _, value := range x.X[len(iris)*size:] {
		cp[len(iris)].Measures = append(cp[len(iris)].Measures, value)
	}
	I := set.ByName["i"]
	for i := range cp {
		cp[i].Embedding = I.X[i*width : (i+1)*width]
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
