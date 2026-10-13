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
