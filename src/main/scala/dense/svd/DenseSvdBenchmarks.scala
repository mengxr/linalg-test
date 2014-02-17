package dense.svd

import java.util.Random

import org.apache.mahout.math.{DenseMatrix, SingularValueDecomposition}
import breeze.linalg.{DenseMatrix => BreezeDenseMatrix, svd => breezeSvd, DenseVector => BreezeDenseVector}
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

  var svd: SingularValueDecomposition = _

  override def run() {
    svd = new SingularValueDecomposition(mat)
  }

  override def certificate(): Double = svd.getSingularValues.head
}

class JblasDenseSvdBenchmark extends DenseSvdBenchmark {

  val mat = new DoubleMatrix(m, n)
  for (i <- 0 until m) {
    for (j <- 0 until n) {
      mat.put(i, j, raw(i + j * m))
    }
  }

  var svd: Array[DoubleMatrix] = _

  override def run() {
    svd = Singular.fullSVD(mat)
  }

  override def certificate(): Double = svd(1).toArray.head
}

class BreezeDenseSvdBenchmark extends DenseSvdBenchmark {

  val mat = new BreezeDenseMatrix[Double](m, n)
  for (i <- 0 until m) {
    for (j <- 0 until n) {
      mat.update(i, j, raw(i + j * m))
    }
  }

  var svd: (BreezeDenseMatrix[Double], BreezeDenseVector[Double], BreezeDenseMatrix[Double]) = _

  override def run() {
    svd = breezeSvd.apply(mat)
  }

  override def certificate(): Double = svd._2.toArray.head
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
