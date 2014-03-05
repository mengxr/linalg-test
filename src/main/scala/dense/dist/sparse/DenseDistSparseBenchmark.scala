package dense.dist.sparse

import util.Benchmark
import java.util.Random
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{DenseVector => BDV, VectorBuilder => BVB, squaredDistance => breezeSquaredDistance}
import org.apache.mahout.math.{DenseVector => MDV, SequentialAccessSparseVector => MSV}

/**
 * This is not a fair benchmark for Mahout, becauce Mahout's implementation would use
 * approximation if two vectors are close to orthogonal.
 */
abstract class DenseDistSparseBenchmark extends Benchmark {
  val n = 100000
  val random = new Random(0)
  val arr = new Array[Double](n)
  val sparsity = 0.02
  val indicesBuffer = ArrayBuffer[Int]()
  val valuesBuffer = ArrayBuffer[Double]()
  (0 until n).foreach { i =>
    val x = random.nextDouble()
    if (x < sparsity) {
      indicesBuffer += i
      valuesBuffer += x
      arr.update(i, x + 0.00001 * random.nextDouble())
    } else {
      arr.update(i, 0.00001 * random.nextDouble())
    }
  }
  val indices = indicesBuffer.toArray
  val values = valuesBuffer.toArray
}

class NaiveDenseDistSparseBenchmark extends DenseDistSparseBenchmark {

  var dist: Double = _

  def run() {
    dist = 0.0
    var i = 0
    var j = 0
    var jj = 0
    var diff = 0.0
    while (j < indices.length) {
      jj = indices(j)
      while (i < jj) {
        diff = arr(i)
        dist += diff * diff
        i += 1
      }
      diff = arr(jj) - values(j)
      dist += diff * diff
      i += 1
      j += 1
    }
    while (i < n) {
      diff = arr(i)
      dist += diff * diff
      i += 1
    }
    dist = math.sqrt(dist)
  }

  def certificate(): Double = dist
}

class BreezeDenseDistSparseBenchmark extends DenseDistSparseBenchmark {

  var dist: Double = _

  val dv = new BDV[Double](arr)

  val svBuilder = new BVB[Double](n)
  indices.zip(values).foreach { case (i, x) =>
    svBuilder.add(i, x)
  }
  val sv = svBuilder.toSparseVector

  def run() {
    dist = math.sqrt(breezeSquaredDistance(dv, sv))
  }

  def certificate(): Double = dist
}

class MahoutDenseDistSparseBenchmark extends DenseDistSparseBenchmark {

  var dist: Double = _

  val dv = new MDV(arr)

  val sv = new MSV(n)
  indices.zip(values).foreach { case (i, x) =>
    sv.set(i, x)
  }

  def run() {
    dist = math.sqrt(dv.getDistanceSquared(sv))
  }

  def certificate(): Double = dist
}

object DenseDistSparseBenchmarks extends App {
  val m = 100000
  val numTrials = 1000
  val naive = new NaiveDenseDistSparseBenchmark
  val breeze = new BreezeDenseDistSparseBenchmark
  val mahout = new MahoutDenseDistSparseBenchmark
  for (bench <- Seq(naive, breeze, mahout)) {
    bench.runBenchmark(m, numTrials)
  }
}
