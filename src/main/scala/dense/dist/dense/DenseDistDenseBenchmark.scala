package dense.dist.dense

import util.Benchmark
import java.util.Random
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{DenseVector => BDV, VectorBuilder => BVB, squaredDistance => breezeSquaredDistance}
import org.apache.mahout.math.{DenseVector => MDV, SequentialAccessSparseVector => MSV}

/**
 * This is not a fair benchmark for Mahout, becauce Mahout's implementation would use
 * approximation if two vectors are close to orthogonal.
 */
abstract class DenseDistDenseBenchmark extends Benchmark {
  val n = 100000
  val random = new Random(0)
  val arr1 = new Array[Double](n)
  val arr2 = new Array[Double](n)
  (0 until n).foreach { i =>
    val x = random.nextDouble()
    arr1.update(i, x + 0.0001 * random.nextDouble())
    arr2.update(i, x + 0.0001 * random.nextDouble())
  }
}

class NaiveDenseDistDenseBenchmark extends DenseDistDenseBenchmark {

  var dist: Double = _

  def run() {
    dist = 0.0
    var i = 0
    var diff = 0.0
    while (i < n) {
      diff = arr1(i) - arr2(i)
      dist += diff * diff
      i += 1
    }
    dist = math.sqrt(dist)
  }

  def certificate(): Double = dist
}

class BreezeDenseDistDenseBenchmark extends DenseDistDenseBenchmark {

  var dist: Double = _

  val dv1 = new BDV[Double](arr1)
  val dv2 = new BDV[Double](arr2)

  def run() {
    dist = math.sqrt(breezeSquaredDistance(dv1, dv2))
  }

  def certificate(): Double = dist
}

class MahoutDenseDistDenseBenchmark extends DenseDistDenseBenchmark {

  var dist: Double = _

  val dv1 = new MDV(arr1)
  val dv2 = new MDV(arr2)

  def run() {
    dist = math.sqrt(dv1.getDistanceSquared(dv2))
  }

  def certificate(): Double = dist
}

object DenseDistDenseBenchmarks extends App {
  val m = 100000
  val numTrials = 1000
  val naive = new NaiveDenseDistDenseBenchmark
  val breeze = new BreezeDenseDistDenseBenchmark
  val mahout = new MahoutDenseDistDenseBenchmark
  for (bench <- Seq(naive, breeze, mahout)) {
    bench.runBenchmark(m, numTrials)
  }
}
