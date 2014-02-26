package dense.norm

import java.util.Random

import breeze.linalg.{DenseVector => BreezeDenseVector, norm => breezeNorm}
import org.apache.mahout.math.{DenseVector => MahoutDenseVector}
import no.uib.cipr.matrix.{DenseVector => MtjDenseVector, Vector => MtjVector}

import util.Benchmark

abstract class DenseNormBenchmark extends Benchmark {
  val n = 1000000
  val random = new Random(0)
  val arr = Array.fill(n)(random.nextDouble)
}

class BreezeDenseNormBenchmark extends DenseNormBenchmark {

  val v = new BreezeDenseVector[Double](arr)

  var nrm: Double = _

  def run() {
    nrm = breezeNorm(v)
  }

  def certificate(): Double = nrm
}

class MahoutDenseNormBenchmark extends DenseNormBenchmark {

  val v = new MahoutDenseVector(arr)

  var nrm: Double = _

  def run() {
    // mahout caches the result
    nrm = v.norm(2.0)
  }

  def certificate(): Double = nrm
}

class NaiveDenseNormBenchmark extends DenseNormBenchmark {

  var nrm: Double = _

  def run() {
    nrm = 0.0
    var i = 0
    var a = 0.0
    while (i < arr.length) {
      a = arr(i)
      nrm += a * a
      i += 1
    }
    nrm = math.sqrt(nrm)
  }

  def certificate: Double = nrm
}

class ClosureDenseNormBenchmark extends DenseNormBenchmark {

  var nrm: Double = _

  def foreach[@specialized(Double) T, @specialized(Unit) R](v: Array[T], fn: (T) => R) {
    var i = 0
    while (i < arr.length) {
      fn(v(i))
      i += 1
    }
  }

  def run() {
    nrm = 0.0
    foreach[Double, Unit](arr, x => nrm += x * x)
    nrm = math.sqrt(nrm)
  }

  def certificate: Double = nrm
}

class MtjDenseNormBenchmark extends DenseNormBenchmark {

  val v = new MtjDenseVector(arr)

  var nrm: Double = _

  def run() {
    nrm = v.norm(MtjVector.Norm.Two)
  }

  def certificate(): Double = nrm
}

object DenseNormBenchmarks extends App {
  val m = 1000
  val numTrials = 100
  val breeze = new BreezeDenseNormBenchmark
  val mahout = new MahoutDenseNormBenchmark
  val naive = new NaiveDenseNormBenchmark
  val mtj = new MtjDenseNormBenchmark
  val closure = new ClosureDenseNormBenchmark
  for (bench <- Seq(naive, closure, breeze, mahout, mtj)) {
    bench.runBenchmark(m, numTrials)
  }
}