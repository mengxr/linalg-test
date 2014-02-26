package dense.dot.sparse

import util.Benchmark
import java.util.Random
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{DenseVector => BDV, VectorBuilder => BVB}
import org.apache.mahout.math.{DenseVector => MDV, SequentialAccessSparseVector => MSV}

abstract class DenseDotSparseBenchmark extends Benchmark {
  val n = 1000000
  val random = new Random(0)
  val arr = Array.fill(n)(random.nextDouble())
  val sparsity = 0.05
  val indicesBuffer = ArrayBuffer[Int]()
  val valuesBuffer = ArrayBuffer[Double]()
  (0 until n).foreach { i =>
    val x = random.nextDouble()
    if (x < sparsity) {
      indicesBuffer += i
      valuesBuffer += x
    }
  }
  val indices = indicesBuffer.toArray
  val values = valuesBuffer.toArray
}

class NaiveDenseDotSparseBenchmark extends DenseDotSparseBenchmark {

  var dot: Double = _

  def run() {
    dot = 0.0
    var i = 0
    while (i < indices.length) {
      dot += arr(indices(i)) * values(i)
      i += 1
    }
  }

  def certificate(): Double = dot
}

class BreezeDenseDotSparseBenchmark extends DenseDotSparseBenchmark {

  var dot: Double = _

  val dv = new BDV[Double](arr)

  val svBuilder = new BVB[Double](n)
  indices.zip(values).foreach { case (i, x) =>
    svBuilder.add(i, x)
  }
  val sv = svBuilder.toSparseVector

  def run() {
    dot = dv.dot(sv)
  }

  def certificate(): Double = dot
}

class MahoutDenseDotSparseBenchmark extends DenseDotSparseBenchmark {

  var dot: Double = _

  val dv = new MDV(arr)

  val sv = new MSV(n)
  indices.zip(values).foreach { case (i, x) =>
    sv.set(i, x)
  }

  def run() {
    dot = dv.dot(sv)
  }

  def certificate(): Double = dot
}

object DenseDotSparseBenchmarks extends App {
  val m = 100000
  val numTrials = 1000
  val naive = new NaiveDenseDotSparseBenchmark
  val breeze = new BreezeDenseDotSparseBenchmark
  val mahout = new MahoutDenseDotSparseBenchmark
  for (bench <- Seq(breeze, naive, mahout)) {
    bench.runBenchmark(m, numTrials)
  }
}