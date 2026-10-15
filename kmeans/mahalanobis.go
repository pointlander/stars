package kmeans

import (
	"fmt"
	"math"
)

// Covariance returns the sample covariance of the rows in data.
// The divisor is n-1. Each row must have the same length.
func Covariance(data [][]float64) ([][]float64, error) {
	n := len(data)
	if n < 2 {
		return nil, fmt.Errorf("covariance needs at least two observations")
	}
	width := len(data[0])
	if width == 0 {
		return nil, fmt.Errorf("observations are empty")
	}
	mean := make([]float64, width)
	for _, row := range data {
		if len(row) != width {
			return nil, fmt.Errorf("observations have different lengths")
		}
		for j, value := range row {
			mean[j] += value
		}
	}
	invN := 1 / float64(n)
	for j := range mean {
		mean[j] *= invN
	}
	cov := make([][]float64, width)
	for i := range cov {
		cov[i] = make([]float64, width)
	}
	scale := 1 / float64(n-1)
	for _, row := range data {
		for i := 0; i < width; i++ {
			di := row[i] - mean[i]
			for j := 0; j <= i; j++ {
				cov[i][j] += di * (row[j] - mean[j])
			}
		}
	}
	for i := 0; i < width; i++ {
		for j := 0; j <= i; j++ {
			cov[i][j] *= scale
			cov[j][i] = cov[i][j]
		}
	}
	return cov, nil
}

// InvertSymmetric inverts a symmetric positive definite matrix.
// The inverse is what MahalanobisDistance expects as inverseCovariance.
func InvertSymmetric(matrix [][]float64) ([][]float64, error) {
	n := len(matrix)
	if n == 0 {
		return nil, fmt.Errorf("empty matrix")
	}
	for i := range matrix {
		if len(matrix[i]) != n {
			return nil, fmt.Errorf("matrix is not square")
		}
	}
	lower, err := cholesky(matrix)
	if err != nil {
		return nil, err
	}
	// Each column of the inverse solves A x = e_col. Columns are stored as
	// rows first, then transposed into the returned matrix.
	inv := make([][]float64, n)
	y := make([]float64, n)
	x := make([]float64, n)
	for col := 0; col < n; col++ {
		for i := 0; i < n; i++ {
			sum := 0.0
			if i == col {
				sum = 1
			}
			for k := 0; k < i; k++ {
				sum -= lower[i][k] * y[k]
			}
			y[i] = sum / lower[i][i]
		}
		for i := n - 1; i >= 0; i-- {
			sum := y[i]
			for k := i + 1; k < n; k++ {
				sum -= lower[k][i] * x[k]
			}
			x[i] = sum / lower[i][i]
		}
		inv[col] = append([]float64(nil), x...)
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			inv[i][j], inv[j][i] = inv[j][i], inv[i][j]
		}
	}
	return inv, nil
}

// MahalanobisFromData estimates the sample covariance of data and returns the
// squared Mahalanobis distance used by Kmeans.
func MahalanobisFromData(data [][]float64) (DistanceFunction, error) {
	cov, err := Covariance(data)
	if err != nil {
		return nil, err
	}
	inv, err := InvertSymmetric(cov)
	if err != nil {
		return nil, err
	}
	return SquaredMahalanobisDistance(inv), nil
}

// cholesky returns the lower factor L of a symmetric positive definite matrix,
// with matrix = L L^T. Only the lower triangle of matrix is read.
func cholesky(matrix [][]float64) ([][]float64, error) {
	n := len(matrix)
	lower := make([][]float64, n)
	for i := range lower {
		lower[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := matrix[i][j]
			for k := 0; k < j; k++ {
				sum -= lower[i][k] * lower[j][k]
			}
			if i == j {
				if sum <= 0 || math.IsNaN(sum) {
					return nil, fmt.Errorf("matrix is not positive definite")
				}
				lower[i][j] = math.Sqrt(sum)
				continue
			}
			lower[i][j] = sum / lower[j][j]
		}
	}
	return lower, nil
}

// MahalanobisDistance returns sqrt((x-y)^T inverseCovariance (x-y)).
func MahalanobisDistance(firstVector, secondVector []float64, inverseCovariance [][]float64) (float64, error) {
	squared, err := mahalanobisSquared(firstVector, secondVector, inverseCovariance)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(squared), nil
}

// SquaredMahalanobisDistance binds an inverse covariance into a DistanceFunction.
// The function returns (x-y)^T inverseCovariance (x-y). Kmeans takes a square
// root of the distance inside near, matching SquaredEuclideanDistance, so this
// squared form is the one to pass to Kmeans.
func SquaredMahalanobisDistance(inverseCovariance [][]float64) DistanceFunction {
	return func(firstVector, secondVector []float64) (float64, error) {
		return mahalanobisSquared(firstVector, secondVector, inverseCovariance)
	}
}

func mahalanobisSquared(firstVector, secondVector []float64, inverseCovariance [][]float64) (float64, error) {
	if len(firstVector) != len(secondVector) {
		return 0, fmt.Errorf("vectors have different lengths")
	}
	n := len(firstVector)
	if len(inverseCovariance) != n {
		return 0, fmt.Errorf("inverse covariance has the wrong size")
	}
	diff := make([]float64, n)
	for i := range diff {
		diff[i] = firstVector[i] - secondVector[i]
	}
	squared := 0.0
	for i := 0; i < n; i++ {
		if len(inverseCovariance[i]) != n {
			return 0, fmt.Errorf("inverse covariance is not square")
		}
		row := 0.0
		for j := 0; j < n; j++ {
			row += inverseCovariance[i][j] * diff[j]
		}
		squared += diff[i] * row
	}
	if squared < 0 {
		if squared > -1e-9 {
			return 0, nil
		}
		return 0, fmt.Errorf("inverse covariance is not positive semidefinite")
	}
	return squared, nil
}
