package dense.svd

import java.util.Random

import org.apache.mahout.math.{DenseMatrix, SingularValueDecomposition}
import breeze.linalg.{DenseMatrix => BreezeDenseMatrix, LinearAlgebra}
import org.jblas.{Singular, DoubleMatrix}

import util.Benchmark

abstract class DenseSvdBenchmark extends Benchmark {
  val m = 500
  val n = 500
  val random = new Random(0)
  val raw = Array.fill(m * n)(random.nextDouble())
}

class MahoutDenseSvdBenchmark extends DenseSvdBenchmark {

  val mat = new DenseMatrix(m, n)
  for (i <- 0 until m) {
    for (j <- 0 until n) {
      mat.set(i, j, raw(i + j * m))
    }
  }

  override def run() {
    new SingularValueDecomposition(mat)
  }
}

class JblasDenseSvdBenchmark extends DenseSvdBenchmark {

  val mat = new DoubleMatrix(m, n)
  for (i <- 0 until m) {
    for (j <- 0 until n) {
      mat.put(i, j, raw(i + j * m))
    }
  }

  override def run() {
    Singular.fullSVD(mat)
  }
}

class BreezeDenseSvdBenchmark extends DenseSvdBenchmark {

  val mat = new BreezeDenseMatrix[Double](m, n)
  for (i <- 0 until m) {
    for (j <- 0 until n) {
      mat.update(i, j, raw(i + j * m))
    }
  }

  override def run() {
    LinearAlgebra.svd(mat)
  }
}

object DenseSvdBenchmarks extends App {

  val n = 10
  val numTrials = 2

  val mahout = new MahoutDenseSvdBenchmark
  mahout.runBenchmark(n, numTrials)
  val breeze = new BreezeDenseSvdBenchmark
  breeze.runBenchmark(n, numTrials)
  val jblas = new JblasDenseSvdBenchmark
  jblas.runBenchmark(n, numTrials)
}
