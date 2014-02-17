package sparse.times.dense

import java.util.Random

import util.Benchmark
import org.apache.mahout.math.{SparseMatrix => MahoutSparseMatrix, DenseMatrix => MahoutDenseMatrix, Matrix => MahoutMatrix}
import breeze.linalg.{CSCMatrix, DenseMatrix => BreezeDenseMatrix}

abstract class SparseMatrixTimesDenseMatrixBenchmark extends Benchmark {
  val m = 5000
  val n = 500
  val p = 100
  val random = new Random(0)
  val rawA = new Array[Double](m*n)
  val rawB = new Array[Double](n*p)
  val sparsity = 0.01
  var nnz = 0
  for (i <- 0 until m*n) {
    val x = random.nextDouble()
    if (x < sparsity) {
      rawA.update(i, x)
      nnz += 1
    }
  }
  for (i <- 0 until n*p) {
    rawB.update(i, random.nextDouble())
  }
}

class BreezeSparseMatrixTimesDenseMatrixBenchmark extends SparseMatrixTimesDenseMatrixBenchmark {
  val builder = new CSCMatrix.Builder[Double](rows = m, cols = n)
  for (i <- 0 until m) {
    for (j <- 0 until n) {
      val x = rawA(i+j*m)
      if (x != 0.0d) builder.add(i, j, x)
    }
  }
  val sparseA = builder.result()
  val denseB = new BreezeDenseMatrix[Double](n, p)
  for (i <- 0 until n) {
    for (j <- 0 until p) {
      denseB.update(i, j, rawB(i+j*n))
    }
  }

  var C: BreezeDenseMatrix[Double] = _

  override def run() {
    C = sparseA * denseB
  }

  override def certificate(): Double = C(0, 0)
}

class MahoutSparseMatrixTimesDenseMatrixBenchmark extends SparseMatrixTimesDenseMatrixBenchmark {
  val denseA = new MahoutDenseMatrix(m, n)
  for(i <- 0 until m) {
    for(j <- 0 until n) {
      denseA.set(i, j, rawA(i+j*m))
    }
  }
  val sparseA = new MahoutSparseMatrix(m, n)
  sparseA.assign(denseA)
  val denseB = new MahoutDenseMatrix(n, p)
  for (i <- 0 until n) {
    for (j <- 0 until p) {
      denseB.set(i, j, rawB(i+j*n))
    }
  }

  var C: MahoutMatrix = _
  override def run() {
    C = sparseA.times(denseB)
  }

  override def certificate(): Double = C.get(0, 0)
}

class NaiveSparseMatrixTimesDenseMatrixBenchmark extends SparseMatrixTimesDenseMatrixBenchmark {

  val ii = new Array[Int](m+1)
  val ij = new Array[Int](nnz)
  val ia = new Array[Double](nnz)

  var idx = 0
  var i = 0
  var j = 0
  while (i < m) {
    ii.update(i, idx)
    j = 0
    while (j < n) {
      val x = rawA(i+j*m)
      if (x != 0.0d) {
        ij.update(idx, j)
        ia.update(idx, x)
        idx += 1
      }
      j += 1
    }
    i += 1
  }
  ii.update(m, nnz)

  val C = new Array[Double](m*p)

  override def run {
    j = 0
    while (j < p) {
      i = 0
      while (i < m) {
        var sum = 0.0d
        idx = ii(i)
        while (idx < ii(i+1)) {
          sum += ia(idx) * rawB(ij(idx) + j*n)
          idx += 1
        }
        C.update(i + j*m, sum)
        i += 1
      }
      j += 1
    }
  }

  override def certificate(): Double = C(0)
}

object SparseMatrixTimesDenseMatrixBenchmarks extends App {

  val n = 5
  val numTrials = 1

  val breeze = new BreezeSparseMatrixTimesDenseMatrixBenchmark
  breeze.runBenchmark(n, numTrials)
  val mahout = new MahoutSparseMatrixTimesDenseMatrixBenchmark
  mahout.runBenchmark(n, numTrials)
  val naive = new NaiveSparseMatrixTimesDenseMatrixBenchmark
  naive.runBenchmark(n, numTrials)
}
