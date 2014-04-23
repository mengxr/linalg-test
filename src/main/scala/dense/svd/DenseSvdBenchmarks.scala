package dense.svd

import java.util.Random

import breeze.linalg.{DenseMatrix => BreezeDenseMatrix, DenseVector => BreezeDenseVector, svd => breezeSvd}
import no.uib.cipr.matrix.{DenseMatrix => MtjDenseMatrix, SVD => MtjSvd}
import org.apache.commons.math3.linear.{Array2DRowRealMatrix => CommonsDenseMatrix, SingularValueDecomposition => CommonsSvd}
import org.apache.mahout.math.{DenseMatrix, SingularValueDecomposition}
import org.jblas.{DoubleMatrix, Singular}
import util.Benchmark
import org.apache.log4j.{Level, Logger}

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

class CommonsDenseSvdBenchmark extends DenseSvdBenchmark {

  val mat = new CommonsDenseMatrix(m, n)
  for (i <- 0 until m) {
    for (j <- 0 until n) {
      mat.setEntry(i, j, raw(i + j * m))
    }
  }

  var svd: CommonsSvd = _

  override def run() {
    svd = new CommonsSvd(mat)
  }

  override def certificate(): Double = svd.getSingularValues.head
}

class MtjDenseSvdBenchmark extends DenseSvdBenchmark {

  val mat = new MtjDenseMatrix(m, n)
  for (i <- 0 until m) {
    for (j <- 0 until n) {
      mat.set(i, j, raw(i + j * m))
    }
  }

  var svd: MtjSvd = _

  override def run() {
    svd = MtjSvd.factorize(mat)
  }

  override def certificate(): Double = svd.getS.head
}

object DenseSvdBenchmarks extends App {

  Logger.getLogger("com.github.fommil.netlib").setLevel(Level.ALL)
  Logger.getLogger("com.github.fommil.jniloader").setLevel(Level.ALL)

  val n = 10
  val numTrials = 2

  val jblas = new JblasDenseSvdBenchmark
  val breeze = new BreezeDenseSvdBenchmark
  val mtj = new MtjDenseSvdBenchmark
  val mahout = new MahoutDenseSvdBenchmark
  val commons = new CommonsDenseSvdBenchmark

  for (bench <- Seq(jblas, breeze, mtj, mahout, commons)) {
    bench.runBenchmark(n, 2)
  }
}
